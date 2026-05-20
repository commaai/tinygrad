import argparse, os, hashlib, functools
from typing import Iterator, Callable
from tinygrad.helpers import getenv, DEBUG, round_up, Timing, tqdm, fetch, ceildiv
from extra.hevc.hevc import HevcParser, parse_hevc_file_headers, untile_nv12, to_bgr, nv_gpu
from tinygrad import Tensor, dtypes, Device, Variable, TinyJit

# rounds up hevc input data to 32 bytes, so more optimal kernels can be generated
HEVC_ROUNDUP = getenv("DATA_ROUNDUP", 32)

@functools.cache
def _hevc_jitted_decoder(out_image_size:tuple[int, int], max_hist:int, inplace:bool, opaque_len:int):
  def hevc_decode_frame(pos:Variable, hevc_tensor:Tensor, offset:Variable, sz:Variable, opaque:Tensor, i:Variable, *hist:Tensor, outbuf:Tensor|None=None):
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

  decode_jit = _hevc_jitted_decoder(out_image_size, max_hist, preallocated_outputs is not None, opaque.shape[0])
  history = history or [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device="NV").contiguous().realize() for _ in range(max_hist)]
  assert len(history) == max_hist, f"history length {len(history)} does not match max_hist {max_hist}"

  for i, (offset, sz, frame_pos, _, is_hist) in enumerate(frame_info):
    history = history[-max_hist:] if max_hist > 0 else []
    img = decode_jit(v_pos.bind(frame_pos), hevc_tensor, v_offset.bind(offset), v_sz.bind(ceildiv(sz, HEVC_ROUNDUP)),
                     opaque, v_i.bind(i), *history, outbuf=preallocated_outputs[i] if preallocated_outputs else None)
    res = preallocated_outputs[i] if preallocated_outputs else img.clone().realize()
    if is_hist: history.append(res)
    yield res

class HevcPacketDecoder:
  def __init__(self, header:bytes=b"", device="NV"):
    self.device, self.parser, self.history = device, HevcParser(header, device=device), []
    self.w, self.h, self.luma_w, self.luma_h, self.chroma_off = self.parser.dimensions()
    self.out_image_size = self.luma_h + (self.luma_h + 1) // 2, round_up(self.luma_w, 64)
    self.buf_slot, self.packet_bufs, self.opaque_bufs = 0, [], []
    self.out_bufs = [Tensor.empty(*self.out_image_size, dtype=dtypes.uint8, device=device).contiguous().realize() for _ in range(16)]
    self.out_rawbufs = [x._buffer() for x in self.out_bufs]

  def _copy_to_slot(self, bufs, slot:int, dat:bytes):
    while len(bufs) <= slot: bufs.append(None)
    if bufs[slot] is None or bufs[slot].numel() < len(dat):
      bufs[slot] = Tensor.empty(round_up(len(dat), 0x100), dtype=dtypes.uint8, device=self.device).contiguous().realize()
    rawbuf = bufs[slot]._buffer()
    rawbuf.allocator._copyin(rawbuf._buf, memoryview(dat))
    return bufs[slot], rawbuf

  def Decode(self, packet) -> list[Tensor]:
    if not packet: return []
    ctx, frame_info = self.parser.parse(packet, tensor=False)
    if not frame_info: return []
    slot = self.buf_slot % 16
    self.buf_slot += 1
    hevc_tensor, hevc_rawbuf = self._copy_to_slot(self.packet_bufs, slot, packet)
    opaque_buf, opaque_rawbuf = self._copy_to_slot(self.opaque_bufs, slot, ctx)
    opaque = opaque_buf.reshape(opaque_buf.numel()//self.parser.align_ctx_bytes_size, self.parser.align_ctx_bytes_size)
    ret = []
    for i, (offset, sz, frame_pos, hist_size, is_hist) in enumerate(frame_info):
      self.history = self.history[-hist_size:] if hist_size else []
      frame = self.out_bufs[frame_pos]
      bitstream = hevc_rawbuf if offset == 0 else hevc_tensor[offset:offset+ceildiv(sz, HEVC_ROUNDUP)*HEVC_ROUNDUP]._buffer()
      desc = opaque_rawbuf if i == 0 else opaque[i]._buffer()
      self.out_rawbufs[frame_pos].allocator._encode_decode(self.out_rawbufs[frame_pos]._buf, bitstream._buf, desc._buf,
        [h._buffer()._buf for h in self.history], self.out_image_size, frame_pos)
      ret.append(frame)
      if is_hist: self.history.append(frame)
      if len(self.history) >= self.parser.sps.sps_max_dec_pic_buffering[0]: self.history.pop(0)
    return ret

  def to_rgb(self, frame:Tensor) -> Tensor:
    return to_bgr(frame, self.h, self.w, self.luma_w, self.chroma_off).flip(2).realize()



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
