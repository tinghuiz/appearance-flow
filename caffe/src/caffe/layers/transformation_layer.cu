#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/transformation_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
__global__ void transform_kernel(int N, int C, int bH, int bW, const Dtype* bot,
    int tH, int tW, Dtype* top, Dtype a00, Dtype a01, Dtype a10, Dtype a11,
    Dtype t0, Dtype t1, Dtype s, const Dtype* m) {
  CUDA_KERNEL_LOOP(i, N*C*tW*tH) {
    const int n = i / (C*tW*tH);
    const int c = (i / (tW*tH)) % C;
    const Dtype y = (i / tW) % tH;
    const Dtype x = i % tW;

    const Dtype xx = a00*x + a01*y + t0, yy = a10*x + a11*y + t1;
    if (0 <= yy && yy < bH && 0 <= xx && xx < bW) {
      // Linear interpolation
      int x0 = xx, y0 = yy, x1 = xx+1, y1 = yy+1;
      if (x1 > bW-1) x1 = bW-1;
      if (y1 > bH-1) y1 = bH-1;
      const Dtype wx = x1 - xx, wy = y1 - yy;
      Dtype v = (wx)   * (wy)   * bot[(n*C+c)*bW*bH + y0*bW + x0] +
                (1-wx) * (wy)   * bot[(n*C+c)*bW*bH + y0*bW + x1] +
                (wx)   * (1-wy) * bot[(n*C+c)*bW*bH + y1*bW + x0] +
                (1-wx) * (1-wy) * bot[(n*C+c)*bW*bH + y1*bW + x1];
      top[i] = s * (v - m[c]);
    } else {
      top[i] = 0;
    }
  }
}

template <typename Dtype>
void TransformationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Comptute the transformation parameters
  std::vector<int> Sbot = bottom[0]->shape(), Stop = top[0]->shape();
  const int W  = Stop[Stop.size()-1], H  = Stop[Stop.size()-2];
  const int bW = Sbot[Sbot.size()-1], bH = Sbot[Sbot.size()-2];
  const int N  = top[0]->count() / (W*H*Stop[Stop.size()-3]);
  std::vector<Affine2D> aff = generate(N, bW, bH, W, H);

  // Transform
  for (int i = 0; i < bottom.size(); i++) {
    const int C  = top[i]->shape()[Stop.size()-3];
    // Get the mean
    const Dtype * mean = mean_value_[i]->gpu_data();

    const Dtype * pBot = bottom[i]->gpu_data();
    Dtype * pTop = top[i]->mutable_gpu_data();
    for (int n = 0; n < N; n++) {
      transform_kernel<Dtype><<<CAFFE_GET_BLOCKS(C*H*W), CAFFE_CUDA_NUM_THREADS>>>(  //NOLINT
        1, C, bH, bW, pBot+n*C*bH*bW, H, W, pTop+n*C*H*W, aff[n].a00_,
        aff[n].a01_, aff[n].a10_, aff[n].a11_, aff[n].t0_, aff[n].t1_,
        (Dtype)scale_, mean);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}


INSTANTIATE_LAYER_GPU_FORWARD(TransformationLayer);


}  // namespace caffe
