#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/layers/fast_hdf5_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
typedef uint16_t half;
template<typename T1, typename T2>
__global__ void convertKernel(const T1 * from, T2 * to, int N) {
  CUDA_KERNEL_LOOP(i, N)
    to[i] = from[i];
}
template<typename T1>
__global__ void convertKernel(const T1 * from, half * to, int N) {
  CUDA_KERNEL_LOOP(i, N)
    to[i] = __float2half_rn(from[i]);
}
template<typename T2>
__global__ void convertKernel(const half * from, T2 * to, int N) {
  CUDA_KERNEL_LOOP(i, N)
    to[i] = __half2float(from[i]);
}
template<typename T1, typename T2>
static void convertGPU(const T1 * from, T2 * to, int N) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  convertKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(from, to, N);
  CUDA_POST_KERNEL_CHECK;
}
static int data_size(int type) {
  if (type == UINT8) return 1;
  if (type == FLOAT16) return 2;
  if (type == FLOAT32) return 4;
  LOG(FATAL) << "Unknown data type";
  return 0;
}
template <typename Dtype>
void FastHDF5OutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  size_t datasize = data_size(type_);
  // Convert the data to the cpu_buffer
  char * mutable_gpu_tmp = static_cast<char*>(gpu_tmp_->mutable_gpu_data());
  char * mutable_cpu_tmp = static_cast<char*>(cpu_tmp_->mutable_cpu_data());
  for (int i = 0; i < bottom.size(); i++) {
    char * p_gpu = mutable_gpu_tmp+offset_[i];
    if (type_ == UINT8)
      convertGPU(bottom[i]->gpu_data(), reinterpret_cast<uint8_t*>(p_gpu),
                 bottom[i]->count());
    else if (type_ == FLOAT16)
      convertGPU(bottom[i]->gpu_data(), reinterpret_cast<half*>(p_gpu),
                 bottom[i]->count());
    else if (type_ == FLOAT32)
      convertGPU(bottom[i]->gpu_data(), reinterpret_cast<float*>(p_gpu),
                 bottom[i]->count());
    else
      LOG(FATAL) << "Unknown data type";
    caffe_copy(bottom[i]->count()*datasize, p_gpu, mutable_cpu_tmp+offset_[i]);
  }
  saveData(bottom);
}
template <typename Dtype>
void FastHDF5InputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  char * gpu_tmp = static_cast<char*>(gpu_tmp_->mutable_gpu_data());
  const char * data = reinterpret_cast<const char*>(batch->memory_->cpu_data());
  size_t o = 0;
  for (int i = 0; i < top.size(); i++) {
    int type = batch->type_[i];
    size_t n = top[i]->count() * data_size(type);
    caffe_copy(n, data+o, gpu_tmp);

    if (type == UINT8)
      convertGPU(reinterpret_cast<const uint8_t*>(gpu_tmp),
                 top[i]->mutable_gpu_data(), top[i]->count());
    else if (type == FLOAT16)
      convertGPU(reinterpret_cast<const half*>(gpu_tmp),
                 top[i]->mutable_gpu_data(), top[i]->count());
    else if (type == FLOAT32)
      convertGPU(reinterpret_cast<const float*>(gpu_tmp),
                 top[i]->mutable_gpu_data(), top[i]->count());
    else
      LOG(FATAL) << "No known data type converter";
    o += n;
  }
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(FastHDF5OutputLayer);
INSTANTIATE_LAYER_GPU_FORWARD(FastHDF5InputLayer);


}  // namespace caffe
