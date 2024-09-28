import os, sys, pickle, time
import numpy as np
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "JIT_BATCH_SIZE" not in os.environ: os.environ["JIT_BATCH_SIZE"] = "0"

from tinygrad import fetch, Tensor, TinyJit, Device, Context, GlobalCounters
from tinygrad.helpers import OSX, DEBUG, getenv
from tinygrad.tensor import _from_np_dtype

import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import get_run_onnx   # TODO: port to main tinygrad

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"
OUTPUT = "/tmp/openpilot.pkl"



MODEL_WIDTH, MODEL_HEIGHT = 512, 256

def tensor_arange(end):
    return Tensor([float(i) for i in range(end)])

def tensor_round(tensor):
    return (tensor + 0.5).floor()

def warp_perspective_tinygrad(src, M_inv, dsize):
    h_dst, w_dst = dsize[1], dsize[0]
    h_src, w_src = src.shape[:2]

    x = tensor_arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
    y = tensor_arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
    ones = Tensor.ones_like(x)
    dst_coords = x.reshape((1,-1)).cat(y.reshape((1,-1))).cat(ones.reshape((1,-1)))


    src_coords = M_inv @ dst_coords
    src_coords = src_coords / src_coords[2:3, :]

    x_src = src_coords[0].reshape(h_dst, w_dst)
    y_src = src_coords[1].reshape(h_dst, w_dst)

    x_nearest = tensor_round(x_src).clip(0, w_src - 1).cast('int')
    y_nearest = tensor_round(y_src).clip(0, h_src - 1).cast('int')

    dst = src[y_nearest, x_nearest]
    return dst


def frame_prepare_tinygrad(input_frame, M_inv, M_inv_uv, W, H):
  input_frame = input_frame.flatten()
  y = warp_perspective_tinygrad(input_frame[:H*W].reshape((H,W)), M_inv, (MODEL_WIDTH, MODEL_HEIGHT)).flatten()
  u = warp_perspective_tinygrad(input_frame[H*W::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  v = warp_perspective_tinygrad(input_frame[H*W+1::2].reshape((H//2,W//2)), M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2)).flatten()
  yuv = y.cat(u).cat(v).reshape((1,MODEL_HEIGHT*3//2,MODEL_WIDTH))
  tensor = frames_to_tensor(yuv)
  return tensor


def frames_to_tensor(frames):
  H = (frames.shape[1]*2)//3
  W = frames.shape[2]
  in_img1 = Tensor.zeros((frames.shape[0], 6, H//2, W//2), dtype='uint8').contiguous()

  in_img1[:, 0] = frames[:, 0:H:2, 0::2]
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1





def compile():
  # hack to fix GPU on OSX: max doesn't work on half, see test/external/external_gpu_fail_osx.py
  if OSX:
    from tinygrad.ops import BinaryOps
    from tinygrad.renderer.cstyle import ClangRenderer, CStyleLanguage
    CStyleLanguage.code_for_op[BinaryOps.MAX] = ClangRenderer.code_for_op[BinaryOps.MAX]

  Tensor.no_grad = True
  Tensor.training = False

  onnx_bytes = fetch(OPENPILOT_MODEL)
  onnx_model = onnx.load(onnx_bytes)
  run_onnx = get_run_onnx(onnx_model)
  print("loaded model")

  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: np.float32 for inp in onnx_model.graph.input}
  input_types['input_imgs'] = np.uint8
  input_types['big_input_imgs'] = np.uint8
  Tensor.manual_seed(100)
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}

  # Extra inputs
  new_inputs['new_img'] = Tensor.randn((1208*3//2, 1908), dtype='uint8').mul(8).realize()
  new_inputs['new_big_img'] = Tensor.randn((1208*3//2, 1908), dtype='uint8').mul(8).realize()
  new_inputs['M_inv'] = Tensor.randn((3,3), dtype='uint8').mul(8).realize()
  new_inputs['M_inv_uv'] = Tensor.randn((3,3), dtype='uint8').mul(8).realize()
  new_inputs['M_inv_wide'] = Tensor.randn((3,3), dtype='uint8').mul(8).realize()
  new_inputs['M_inv_uv_wide'] = Tensor.randn((3,3), dtype='uint8').mul(8).realize()


  # Model + pipeline
  def img_pipeline_and_onnx(tensor_dict):
    W, H = 1908, 1208
    tensor_dict['input_imgs'][:,:6] = tensor_dict['input_imgs'][:,6:]
    tensor_dict['input_imgs'][:,6:] = frame_prepare_tinygrad(tensor_dict['new_img'],
                                                               tensor_dict['M_inv'],
                                                               tensor_dict['M_inv_uv'],
                                                               W, H)

    tensor_dict['big_input_imgs'][:,:6] = tensor_dict['big_input_imgs'][:,6:]
    tensor_dict['big_input_imgs'][:,6:] = frame_prepare_tinygrad(tensor_dict['new_big_img'],
                                                                   tensor_dict['M_inv_wide'],
                                                                   tensor_dict['M_inv_uv_wide'],
                                                                   W,H)

    onnx_kwargs = {inp.name:tensor_dict[inp.name] for inp in onnx_model.graph.input}
    return run_onnx(onnx_kwargs)


  full_jit = TinyJit(lambda **kwargs: img_pipeline_and_onnx(kwargs), prune=True)
  for i in range(3):
    GlobalCounters.reset()
    print(f"run {i}")
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      ret = next(iter(full_jit(**new_inputs).values())).cast('float32').numpy()
    if i == 0: test_val = np.copy(ret)
  print(f"captured {len(full_jit.captured.jit_cache)} kernels")
  #np.testing.assert_equal(test_val, ret)
  print("jit run validated")

  with open(OUTPUT, "wb") as f:
    pickle.dump(full_jit, f)
  mdl_sz = os.path.getsize(onnx_bytes)
  pkl_sz = os.path.getsize(OUTPUT)
  print(f"mdl size is {mdl_sz/1e6:.2f}M")
  print(f"pkl size is {pkl_sz/1e6:.2f}M")
  print("**** compile done ****")
  return test_val

def test(test_val=None):
  with open(OUTPUT, "rb") as f:
    run = pickle.load(f)
  Tensor.manual_seed(100)
  new_inputs = {nm:Tensor.randn(*st.shape, dtype=dtype).mul(8).realize() for nm, (st, _, dtype, _) in
                sorted(zip(run.captured.expected_names, run.captured.expected_st_vars_dtype_device))}
  for _ in range(20):
    st = time.perf_counter()
    out = run(**new_inputs)
    mt = time.perf_counter()
    val = out['outputs'].numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")
  print(out, val.shape, val.dtype)
  if test_val is not None: np.testing.assert_equal(test_val, val)
  print("**** test done ****")

if __name__ == "__main__":
  test_val = compile() if not getenv("RUN") else None
  test(test_val)

