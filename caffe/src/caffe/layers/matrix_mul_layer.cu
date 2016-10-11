#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/matrix_mul_layer.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

template<typename Dtype>
void MatrixMulLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count0 = bottom[0]->count();
  int count1 = bottom[1]->count();
  int offset_data_1 = 0, offset_data_0 = 0;
  
  //the first bottom
  const Dtype* bottom_data_0 = bottom[0]->gpu_data();
  //the second bottom  
  const Dtype* bottom_data_1 = bottom[1]->gpu_data();

  Dtype* top_data = top[0]->mutable_gpu_data();
  
  if(use_streams_){
    default_stream = new cudaStream_t;
    cublasGetStream(Caffe::cublas_handle(), default_stream);
    if(streams_need_init_){
      stream_ = new cudaStream_t[max_streams_];
      for(int i = 0; i < max_streams_; i++){
        //create a new stream
        CUDA_CHECK(cudaStreamCreate(&stream_[i]));
      }
      streams_need_init_ = false;
    }
  }
  
  for(int i=0; i<maxch; i++){
    if(use_streams_){
      // set CUBLAS to use stream
      CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), stream_[i % max_streams_]));
    }
    
    //do matrix multiplication
    if(transpose_0){
      if(transpose_1)
        caffe_gpu_gemm(CblasTrans, CblasTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
      else
        caffe_gpu_gemm(CblasTrans, CblasNoTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
    }
    else{
      if(transpose_1)
        caffe_gpu_gemm(CblasNoTrans, CblasTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
      else
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
    }

    top_data+=M*N;
    offset_data_0 +=M*K; offset_data_0 = offset_data_0 % count0;
    offset_data_1 +=K*N; offset_data_1 = offset_data_1 % count1;
  }

  if(use_streams_){
    // set default stream
    CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), *default_stream));
    // Synch streams
    for(int i = 0; i < max_streams_; i++){
      CUDA_CHECK(cudaStreamSynchronize(stream_[i]));
      // CUDA_CHECK(cudaStreamDestroy(stream_[i]));
    }
  }
}

template<typename Dtype>
void MatrixMulLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count0 = bottom[0]->count();
  int count1 = bottom[1]->count();
  
  //all the bottoms 
  if(propagate_down[0]){
    int offset_data_1 = 0, offset_diff_0 = 0;
    
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff_0 = bottom[0]->mutable_gpu_diff();
    
    const Dtype* bottom_data_1 = bottom[1]->gpu_data();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom_diff_0);
    
    bool can_use_stream = use_streams_ && ch_0 == maxch;

    for(int i = 0; i < maxch; i++){
      if(can_use_stream){
        CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), stream_[i % max_streams_]));
      }
      if(transpose_0){
        if(transpose_1)
          caffe_gpu_gemm(CblasTrans, CblasTrans, K, M, N, Dtype(1.0),
                     bottom_data_1 + offset_data_1, top_diff, Dtype(1.0), bottom_diff_0 + offset_diff_0);
        else
          caffe_gpu_gemm(CblasNoTrans, CblasTrans, K, M, N, Dtype(1.0),
                     bottom_data_1 + offset_data_1, top_diff, Dtype(1.0), bottom_diff_0 + offset_diff_0);
      }
      else{
        if(transpose_1)
          caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M, K, N, Dtype(1.0), top_diff,
                     bottom_data_1 + offset_data_1, Dtype(1.0), bottom_diff_0 + offset_diff_0);
        else
          caffe_gpu_gemm(CblasNoTrans, CblasTrans, M, K, N, Dtype(1.0), top_diff,
                     bottom_data_1 + offset_data_1, Dtype(1.0), bottom_diff_0 + offset_diff_0);
      }
      top_diff += M*N;
      offset_diff_0 += M*K; offset_diff_0 = offset_diff_0 % count0;
      offset_data_1 += K*N; offset_data_1 = offset_data_1 % count1;
    }
    if(can_use_stream){
      CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), *default_stream));
      for(int i = 0; i < max_streams_; i++){
        CUDA_CHECK(cudaStreamSynchronize(stream_[i]));
      }
    }
  }
  
  if(propagate_down[1]){
    int offset_data_0 = 0, offset_diff_1 = 0;

    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();
    const Dtype* bottom_data_0 = bottom[0]->gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0.0), bottom_diff_1);
    
    bool can_use_stream = use_streams_ && ch_1 == maxch;
    for(int i = 0; i < maxch; i++){
      if(can_use_stream){
        CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), stream_[i % max_streams_]));
      }
      if(transpose_0){
        if(transpose_1)
          caffe_gpu_gemm(CblasTrans, CblasTrans, N, K, M, Dtype(1.0), 
                        top_diff, bottom_data_0 + offset_data_0, Dtype(1.0), bottom_diff_1 + offset_diff_1);
        else
          caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, K, N, M, Dtype(1.0), bottom_data_0 + offset_data_0,
                        top_diff, Dtype(1.0), bottom_diff_1 + offset_diff_1);
      }
      else{
        if(transpose_1)
          caffe_gpu_gemm(CblasTrans, CblasNoTrans, N, K, M, Dtype(1.0), 
                        top_diff, bottom_data_0 + offset_data_0, Dtype(1.0), bottom_diff_1 + offset_diff_1);
        else
          caffe_gpu_gemm(CblasTrans, CblasNoTrans, K, N, M, Dtype(1.0), bottom_data_0 + offset_data_0,
                        top_diff, Dtype(1.0), bottom_diff_1 + offset_diff_1);
      }
      top_diff += M*N;

      offset_data_0 += M*K; offset_data_0 = offset_data_0 % count0;
      offset_diff_1 += K*N; offset_diff_1 = offset_diff_1 % count1;
    }
    if(can_use_stream){
      CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), *default_stream));
      for(int i = 0; i < max_streams_; i++){
        CUDA_CHECK(cudaStreamSynchronize(stream_[i]));
      }
    }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(MatrixMulLayer);
}
