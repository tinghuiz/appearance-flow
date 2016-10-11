#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

extern __shared__ char blur_kernel_cache[];
template <typename T, int MR>
__global__ void tentx_k(const T * in, int W, int H, int D, T * out, T R) {
  assert(blockDim.z == 1);
  T * cache = reinterpret_cast<T*>(blur_kernel_cache);
  // Read an image patch into local memory
  const int bW = blockDim.x, bH = blockDim.y;
  const int x0 = bW *blockIdx.x, y0 = bH *blockIdx.y, z = blockIdx.z;
  const int dx = threadIdx.x, dy = threadIdx.y, pW = bW + 2*MR, pH = bH;
  for (int i = dx + bW *dy; i < pW*pH; i += bW * bH) {
    int x = x0 + (i % pW) - MR, y = y0 + (i / pW);
    if (x <  0) x = 0;
    if (x >= W) x = W-1;
    if (y <  0) y = 0;
    if (y >= H) y = H-1;
    cache[i] = in[x+y*W+z*W*H];
  }
  __syncthreads();
  // Aplpy the 1d filter
  int x = x0 + dx, y = y0 + dy;
  if (x < W && y < H) {
    T s = R * cache[(dx+MR) + pW * dy];
    for (int r = 1; r <= MR; r++)
      s += (R-r) * (cache[(dx+r+MR) + pW*dy] + cache[(dx-r+MR) + pW*dy]);
    T norm = ((2*MR+1)*R - MR*(MR+1));  // Closed from for the normalization
    out[x+y*W+z*W*H] = s / norm;
  }
}
template <typename T, int MR>
__global__ void tenty_k(const T * in, int W, int H, int D, T * out, T R) {
  assert(blockDim.z == 1);
  T * cache = reinterpret_cast<T*>(blur_kernel_cache);
  // Read an image patch into local memory
  const int bW = blockDim.x, bH = blockDim.y;
  const int x0 = bW *blockIdx.x, y0 = bH *blockIdx.y, z = blockIdx.z;
  const int dx = threadIdx.x, dy = threadIdx.y, pW = bW, pH = bH + 2*MR;
  for (int i = dx + bW *dy; i < pW*pH; i += bW * bH) {
    int x = x0 + (i % pW), y = y0 + (i / pW) - MR;
    if (x <  0) x = 0;
    if (x >= W) x = W-1;
    if (y <  0) y = 0;
    if (y >= H) y = H-1;
    cache[i] = in[x+y*W+z*W*H];
  }
  __syncthreads();
  // Aplpy the 1d filter
  int x = x0 + dx, y = y0 + dy;
  if (x < W && y < H) {
    T s = R * cache[dx + pW * (dy+MR)];
    for (int r = 1; r <= MR; r++)
      s += (R-r) * (cache[dx + pW * (dy-r+MR)] + cache[dx + pW * (dy+r+MR)]);
    T norm = ((2*MR+1)*R - MR*(MR+1));  // Closed from for the normalization
    out[x+y*W+z*W*H] = s / norm;
  }
}
static int getBS(int i, int MAX_BS = 64) {
  // Round up the the next power of two
  int r = 1;
  for (; r < i && 2*r <= MAX_BS; r *= 2) {}
  return std::min(r, MAX_BS);
}
template <typename T>
static void tentx_gpu(const T* in, int W, int H, int D, T* out, T R) {
  const int BX = getBS(W);
  const int BY = getBS(H, 1024/BX);
  int MR = R;
  const int NS = (BX+2*MR)*BY*sizeof(T);
  if (MR == 0) caffe_copy(W*H*D, in, out);
  // NOLINT_NEXT_LINE
#define CALL_K(X) else if (MR == X) tentx_k<T,X><<<dim3((W-1)/BX+1,(H-1)/BY+1,D),dim3(BX,BY,1),NS>>>(in, W, H, D, out, R)
  CALL_K(1);
  CALL_K(2);
  CALL_K(3);
  CALL_K(4);
  CALL_K(5);
  CALL_K(6);
  CALL_K(7);
  CALL_K(8);
  CALL_K(9);
  CALL_K(10);
  // NOLINT_NEXT_LINE
  else {
    LOG(WARNING) << "Filter radius too large, applying a filter of radius 10";
    // NOLINT_NEXT_LINE
    tentx_k<T, 10><<<dim3((W-1)/BX+1, (H-1)/BY+1, D), dim3(BX, BY, 1), NS>>>(in, W, H, D, out, R);
  }
#undef CALL_K
  CUDA_POST_KERNEL_CHECK;
}
template <typename T>
static void tenty_gpu(const T* in, int W, int H, int D, T* out, T R) {
  const int BX = getBS(W);
  const int BY = getBS(H, 1024/BX);
  int MR = R;
  const int NS = BX*(BY+2*MR)*sizeof(T);
  if (MR < 1) caffe_copy(W*H*D, in, out);
  // NOLINT_NEXT_LINE
#define CALL_K(X) else if (MR == X) tenty_k<T,X><<<dim3((W-1)/BX+1,(H-1)/BY+1,D),dim3(BX,BY,1),NS>>>(in, W, H, D, out, R)
  CALL_K(1);
  CALL_K(2);
  CALL_K(3);
  CALL_K(4);
  CALL_K(5);
  CALL_K(6);
  CALL_K(7);
  CALL_K(8);
  CALL_K(9);
  CALL_K(10);
  // NOLINT_NEXT_LINE
  else {
    LOG(WARNING) << "Filter radius too large, applying a filter of radius 10";
    // NOLINT_NEXT_LINE
    tenty_k<T, 10><<<dim3((W-1)/BX+1, (H-1)/BY+1, D), dim3(BX, BY, 1), NS>>>(in, W, H, D, out, R);
  }
#undef CALL_K
  CUDA_POST_KERNEL_CHECK;
}
template <typename T>
__global__ void lerpx_k(const T * in, int W, int H, int D, int bW,
                        float s, float c, T * output) {
  CUDA_KERNEL_LOOP(i, W*H*D) {
    const int x = i % W;
    const int y = (i / W) % H;
    const int z = (i / W) / H;

    const T o_x = min(max(T(0), x / s + c), T(bW-1));
    const int x0 = o_x;
    const int x1 = min(x0+1, bW-1);
    const T wx = x1 - o_x;

    output[i] =  wx   *in[z*bW*H + y*bW + x0] +
                (1-wx)*in[z*bW*H + y*bW + x1];
  }
}
template <typename T>
__global__ void lerpy_k(const T * in, int W, int H, int D, int bH,
                        float s, float c, T * output) {
  CUDA_KERNEL_LOOP(i, W*H*D) {
    const int x = i % W;
    const int y = (i / W) % H;
    const int z = (i / W) / H;

    const T o_y = min(max(T(0), y / s + c), T(bH-1));
    const int y0 = o_y;
    const int y1 = min(y0+1, bH-1);
    const T wy = y1 - o_y;

    output[i] =  wy   *in[z*W*bH + y0*W + x] +
                (1-wy)*in[z*W*bH + y1*W + x];
  }
}

template <typename T>
void ResizeLayer<T>::Forward_gpu(const vector<Blob<T>*>& bottom,
      const vector<Blob<T>*>& top) {
  const std::vector<int> & S = bottom[0]->shape(), &tS = top[0]->shape();
  const int W = S[S.size()-1], H = S[S.size()-2];
  const int D = bottom[0]->count() / (W*H);
  const int tW = tS[tS.size()-1], tH = tS[tS.size()-2];

  // Apply the blur
  const T * p_bot = bottom[0]->gpu_data();
  T * p_tmp0 = tmp_.mutable_gpu_data();
  T * p_tmp1 = tmp_.mutable_gpu_diff();
  T * p_top = top[0]->mutable_gpu_data();
  // Resize in x
  if (scale_x_ < 1) {
    // If the sampling was perfect (no interpolation later), then a tent filter
    // with radius 1/scale will distribute all bottom values between the top
    // samples
    tentx_gpu(p_bot, W, H, D, p_tmp0, T(1.)/std::max(scale_x_, T(1e-2)));
    p_bot = p_tmp0;
  }
  //  NOLINT_NEXT_LINE
  lerpx_k<<<CAFFE_GET_BLOCKS(tW*H*D), CAFFE_CUDA_NUM_THREADS>>>(p_bot, tW, H, D, W, scale_x_, crop_x_, p_tmp1);
  CUDA_POST_KERNEL_CHECK;

  // Resize in y
  if (scale_y_ < 1) {
    tenty_gpu(p_tmp1, tW, H, D, p_tmp0, T(1.)/std::max(scale_y_, T(1e-2)));
    std::swap(p_tmp1, p_tmp0);
  }
  //  NOLINT_NEXT_LINE
  lerpy_k<<<CAFFE_GET_BLOCKS(tW*H*D), CAFFE_CUDA_NUM_THREADS>>>(p_tmp1, tW, tH, D, H, scale_y_, crop_y_, p_top);
  CUDA_POST_KERNEL_CHECK;
}
template <typename T, int S>
__global__ void unlerpx_k(const T * in, int W, int H, int D, int bW,
                          T s, T c, T * output) {
  CUDA_KERNEL_LOOP(i, bW*H*D) {
    const int x = i % bW;
    const int y = (i / bW) % H;
    const int z = (i / bW) / H;

    const T t_x = x * s - c;
    const int x0 = t_x - s + 0.999;  // Numerically stable ceil (if s<500)
    T sm = 0;
    for (int xx = x0; xx <= x0+S; xx++)
      if (0 <= xx && xx < W) {
        const T o_x = min(max(T(0), xx / s + c), T(bW-1));
        const T w = max(T(1)-fabs(x-o_x), T(0));
        sm += w * in[z*W*H + y*W + xx];
      }
    output[i] = sm;
  }
}
template <typename T, int S>
__global__ void unlerpy_k(const T * in, int W, int H, int D, int bH,
                          T s, T c, T * output) {
  CUDA_KERNEL_LOOP(i, W*bH*D) {
    int x = i % W;
    int y = (i / W) % bH;
    int z = (i / W) / bH;

    const T t_y = y * s - c;
    const int y0 = t_y - s + 0.999;  // Numerically stable ceil (if s<500)
    T sm = 0;
    for (int yy = y0; yy < y0+S; yy++)
      if (0 <= yy && yy < H) {
        const T o_y = min(max(T(0), yy / s + c), T(bH-1));
        const T w = max(T(1)-fabs(y-o_y), T(0));
        sm += w * in[z*W*H + yy*W + x];
      }
    output[i] = sm;
  }
}
template <typename T>
static void unlerpx_gpu(const T * in, int W, int H, int D, int bW,
                        T s, T c, T * out) {
  const int BX = getBS(bW);
  const int BY = getBS(H, 1024/BX);
  int S = ceil(2*s);
  const int NB = CAFFE_GET_BLOCKS(bW*H*D);
  const int NT = CAFFE_CUDA_NUM_THREADS;
  if (S == 0) LOG(FATAL) << "Scale factor cannot be 0";
  // NOLINT_NEXT_LINE[whitespace/operators]
#define CALL_K(X) else if (S == X) unlerpx_k<T,X><<<NB, NT>>>(in, W, H, D, bW, s, c, out)
  CALL_K(1); CALL_K(2); CALL_K(3); CALL_K(4); CALL_K(5);
  CALL_K(6); CALL_K(7); CALL_K(8); CALL_K(9); CALL_K(10);
  CALL_K(11); CALL_K(12); CALL_K(13); CALL_K(14); CALL_K(15);
  CALL_K(16); CALL_K(17); CALL_K(18); CALL_K(19); CALL_K(20);
  // NOLINT_NEXT_LINE[readability/braces]
  else {
    LOG(WARNING) << "Unlerp radius too large, using only 20 elements";
    // NOLINT_NEXT_LINE[whitespace/operators]
    unlerpx_k<T,20><<<NB, NT>>>(in, W, H, D, bW, s, c, out);
  }
#undef CALL_K
  CUDA_POST_KERNEL_CHECK;
}
template <typename T>
static void unlerpy_gpu(const T * in, int W, int H, int D, int bH,
                        T s, T c, T * out) {
  const int BX = getBS(W);
  const int BY = getBS(bH, 1024/BX);
  int S = ceil(2*s);
  const int NB = CAFFE_GET_BLOCKS(W*bH*D);
  const int NT = CAFFE_CUDA_NUM_THREADS;
  if (S == 0) LOG(FATAL) << "Scale factor cannot be 0";
  // NOLINT_NEXT_LINE[whitespace/operators]
#define CALL_K(X) else if (S == X) unlerpy_k<T,X><<<NB, NT>>>(in, W, H, D, bH, s, c, out)
  CALL_K(1); CALL_K(2); CALL_K(3); CALL_K(4); CALL_K(5);
  CALL_K(6); CALL_K(7); CALL_K(8); CALL_K(9); CALL_K(10);
  CALL_K(11); CALL_K(12); CALL_K(13); CALL_K(14); CALL_K(15);
  CALL_K(16); CALL_K(17); CALL_K(18); CALL_K(19); CALL_K(20);
  // NOLINT_NEXT_LINE[readability/braces]
  else {
    LOG(WARNING) << "Unlerp radius too large, using only 20 elements";
    // NOLINT_NEXT_LINE[whitespace/operators]
    unlerpy_k<T,20><<<NB, NT>>>(in, W, H, D, bH, s, c, out);
  }
#undef CALL_K
  CUDA_POST_KERNEL_CHECK;
}
template <typename T>
void ResizeLayer<T>::Backward_gpu(const vector<Blob<T>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<T>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const std::vector<int> & S = bottom[0]->shape(), &tS = top[0]->shape();
  const int W = S[S.size()-1], H = S[S.size()-2];
  const int D = bottom[0]->count() / (W*H);
  const int tW = tS[tS.size()-1], tH = tS[tS.size()-2];

  T * p_bot = bottom[0]->mutable_gpu_diff();
  T * p_tmp0 = tmp_.mutable_gpu_data();
  T * p_tmp1 = tmp_.mutable_gpu_diff();
  const T * p_top = top[0]->gpu_diff();

  // Resize in y
  unlerpy_gpu(p_top, tW, tH, D, H, scale_y_, crop_y_, p_tmp1);
  if (scale_y_ < 1) {
    tenty_gpu(p_tmp1, tW, H, D, p_tmp0, T(1.)/std::max(scale_y_, T(1e-2)));
    std::swap(p_tmp1, p_tmp0);
  }

  // Resize in x
  unlerpx_gpu(p_tmp1, tW, H, D, W, scale_x_, crop_x_, p_bot);
  if (scale_x_ < 1) {
    caffe_copy(W*H*D, p_bot, p_tmp0);
    tentx_gpu(p_tmp0, W, H, D, p_bot, T(1.)/std::max(scale_x_, T(1e-2)));
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);
// INSTANTIATE_LAYER_GPU_FORWARD(ResizeLayer);
}  // namespace caffe
