import argparse, os, hashlib, functools
from typing import Iterator
import numpy as np
from tinygrad.helpers import getenv, round_up, Timing, tqdm, fetch, ceildiv
from extra.hevc.hevc import parse_hevc_file_headers, untile_nv12, to_bgr
from tinygrad import Tensor, dtypes, Device, Variable, TinyJit
from tinygrad.uop.ops import UOp, Ops

# rounds up hevc input data to 32 bytes, so more optimal kernels can be generated
HEVC_ROUNDUP = getenv("DATA_ROUNDUP", 32)

@functools.cache
def _hevc_jitted_decoder(out_image_size:tuple[int, int], max_hist:int, inplace:bool):
  def hevc_decode_frame(pos:Variable, hevc_tensor:Tensor, offset:Variable, sz:Variable, opaque:Tensor, i:Variable,
                        *hist:Tensor, outbuf:Tensor|None=None):
    x = hevc_tensor[offset:offset+sz*HEVC_ROUNDUP].decode_hevc_frame(pos, out_image_size, opaque[i], hist).realize()
    if outbuf is not None: outbuf.assign(x).realize()
    return x
  return TinyJit(hevc_decode_frame)

def hevc_decode(hevc_tensor:Tensor, opaque:Tensor, frame_info:list, luma_h:int, luma_w:int,
                history:list[Tensor]|None=None, preallocated_outputs:list[Tensor]|None=None, warmup=False) -> Iterator[Tensor]:
  out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)
  max_hist = max((hs for _, _, _, hs, _ in frame_info), default=0)

  v_pos = Variable("pos", 0, max_hist + 1)
  v_offset = Variable("offset", 0, hevc_tensor.numel()-1)
  v_sz = Variable("sz", 1, ceildiv(hevc_tensor.numel(), HEVC_ROUNDUP))
  v_i = Variable("i", 0, len(frame_info)-1)

  decode_jit = _hevc_jitted_decoder(out_image_size, max_hist, preallocated_outputs is not None)
  history = history or [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device="NV").contiguous().realize() for _ in range(max_hist)]
  assert len(history) == max_hist, f"history length {len(history)} does not match max_hist {max_hist}"

  for i, (offset, sz, frame_pos, _, is_hist) in enumerate(frame_info):
    history = history[-max_hist:] if max_hist > 0 else []
    img = decode_jit(v_pos.bind(frame_pos), hevc_tensor, v_offset.bind(offset), v_sz.bind(ceildiv(sz, HEVC_ROUNDUP)),
                     opaque, v_i.bind(i), *history, outbuf=preallocated_outputs[i] if preallocated_outputs else None)
    res = preallocated_outputs[i] if preallocated_outputs else img.clone().realize()
    if is_hist: history.append(res)
    yield res

def hevc_preload_packets(dat:bytes, frame_info:list, device:str="NV") -> list[Tensor]:
  dat_np = np.frombuffer(dat, dtype=np.uint8)
  packets = []
  for offset, sz, _, _, _ in frame_info:
    packet_size = ceildiv(sz, HEVC_ROUNDUP) * HEVC_ROUNDUP
    packet = np.zeros(packet_size, dtype=np.uint8)
    packet[:sz] = dat_np[offset:offset+sz]
    packets.append(Tensor(packet, dtype=dtypes.uint8, device=device).contiguous().realize())
  return packets

def _decode_hevc_frame_into(src:Tensor, pos:Variable, out:Tensor, state:Tensor, hist:list[Tensor], out_image_size:tuple[int, int]) -> Tensor:
  srcs = (out, src.contiguous(), state.contiguous(), *[x.contiguous() for x in hist])
  fn = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(pos.src[0], *[UOp.const(dtypes.int, s) for s in out_image_size]), arg="encdec")
  return Tensor(out.uop.after(fn.call(*[s.uop for s in srcs], pos)))

def hevc_decode_preloaded(packets:list[Tensor], opaque:Tensor, frame_info:list, luma_h:int, luma_w:int,
                          history:list[Tensor]|None=None, outputs:list[Tensor]|None=None,
                          device:str="NV") -> Iterator[Tensor]:
  out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)
  max_hist = max((hs for _, _, _, hs, _ in frame_info), default=0)
  opaque = opaque.contiguous().realize()
  history = history or [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device=device).contiguous().realize() for _ in range(max_hist)]
  outputs = outputs or [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device=device).contiguous().realize() for _ in range(max_hist + 1)]

  assert len(history) == max_hist, f"history length {len(history)} does not match max_hist {max_hist}"
  assert len(packets) >= len(frame_info), f"packet count {len(packets)} is less than frame count {len(frame_info)}"
  assert len(outputs) >= len(frame_info) or len(outputs) >= max_hist + 1, f"not enough output buffers: {len(outputs)}"

  opaque_buf = opaque.uop.buffer.ensure_allocated()._buf
  packet_bufs = [packet.contiguous().realize().uop.buffer.ensure_allocated()._buf for packet in packets]
  output_bufs = [output.contiguous().realize().uop.buffer.ensure_allocated()._buf for output in outputs]
  history_bufs = [hist.contiguous().realize().uop.buffer.ensure_allocated()._buf for hist in history]
  desc_stride = opaque.shape[1]
  alloc = Device[device].allocator

  for i, (_, _, frame_pos, _, is_hist) in enumerate(frame_info):
    history = history[-max_hist:] if max_hist > 0 else []
    history_bufs = history_bufs[-max_hist:] if max_hist > 0 else []
    output_idx = i if len(outputs) >= len(frame_info) else frame_pos
    alloc._encode_decode(output_bufs[output_idx], packet_bufs[i], opaque_buf.offset(i*desc_stride), history_bufs, out_image_size, frame_pos)
    res = outputs[output_idx]
    if is_hist:
      history.append(res)
      history_bufs.append(output_bufs[output_idx])
    yield res

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str, default="")
  parser.add_argument("--output_dir", type=str, default="extra/hevc/out")
  args = parser.parse_args()

  if args.input_file == "":
    url = "https://github.com/haraschax/filedump/raw/09a497959f7fa6fd8dba501a25f2cdb3a41ecb12/comma_video.hevc"
    hevc_tensor = Tensor.from_url(url, device="CPU")
  else:
    hevc_tensor = Tensor.empty(os.stat(args.input_file).st_size, dtype=dtypes.uint8, device=f"disk:{args.input_file}").to("CPU")

  dat = bytes(hevc_tensor.data())
  dat_hash = hashlib.md5(dat).hexdigest()

  with Timing("prep infos: "):
    opaque, frame_info, w, h, luma_w, luma_h, chroma_off = parse_hevc_file_headers(dat)

  frame_info = frame_info[:getenv("MAX_FRAMES", len(frame_info))]

  # move all needed data to gpu
  with Timing("copy to gpu: "):
    opaque_nv = opaque.to("NV").contiguous().realize()
    hevc_tensor = hevc_tensor.to("NV")

  out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)

  # preallocate output/hist buffers
  max_hist = max((hs for _, _, _, hs, _ in frame_info), default=0)
  hist = [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device="NV").contiguous().realize() for _ in range(max_hist)]
  out_images = [Tensor.zeros(*out_image_size, dtype=dtypes.uint8, device="NV").contiguous().realize() for _ in range(len(frame_info))]

  # warmup decode
  _ = list(hevc_decode(hevc_tensor, opaque_nv, frame_info[:3], luma_h, luma_w, history=hist, preallocated_outputs=out_images))
  Device.default.synchronize()

  # decode all frames using the iterator
  tm = Timing("decoding whole file: ", on_exit=(lambda et: f", {len(frame_info)} frames, {len(frame_info)/(et/1e9):.2f} fps"))
  with tm:
    images = list(hevc_decode(hevc_tensor, opaque_nv, frame_info, luma_h, luma_w, history=hist, preallocated_outputs=out_images))
    Device.default.synchronize()

  fps = len(frame_info)/(tm.et/1e9)
  assert fps >= getenv("ASSERT_FPS", 0), f"HEVC decode too slow: {fps:.2f} fps"

  # validation
  if getenv("VALIDATE", 0):
    import pickle
    if dat_hash == "b813bfdbec194fd17fdf0e3ceb8cea1c":
      url = "https://github.com/nimlgen/hevc_validate_set/raw/refs/heads/main/decoded_frames_b813bfdbec194fd17fdf0e3ceb8cea1c.pkl"
      decoded_frames = pickle.load(fetch(url).open("rb"))
    else: decoded_frames = pickle.load(open(f"extra/hevc/decoded_frames_{dat_hash}.pkl", "rb"))
  else: import cv2

  for i, img in tqdm(enumerate(images)):
    if getenv("VALIDATE", 0):
      if i < len(decoded_frames) and len(decoded_frames[i]) > 0:
        img = untile_nv12(img, h, w, luma_w, chroma_off).realize()
        assert img.data() == decoded_frames[i], f"Frame {i} does not match reference decoder!"
        print(f"Frame {i} matches reference decoder!")
    else:
      if len(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        img = to_bgr(img, h, w, luma_w, chroma_off).realize()
        cv2.imwrite(f"{args.output_dir}/out_frame_{i:04d}.png", img.numpy())
