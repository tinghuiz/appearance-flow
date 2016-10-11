#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/matrix_mul_layer.hpp"

namespace caffe {

template <typename Dtype>
void MatrixMulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  max_streams_ = this->layer_param_.matrix_mul_param().num_streams();
  use_streams_ = this->layer_param_.matrix_mul_param().use_streams();
  transpose_0 = this->layer_param_.matrix_mul_param().transpose_0();
  transpose_1 = this->layer_param_.matrix_mul_param().transpose_1();
  streams_need_init_ = true;
  LOG(INFO) << "use_streams_: "<< use_streams_;
  LOG(INFO) << "streams_need_init_: "<< streams_need_init_;
  LOG(INFO) << "transpose_0: "<< transpose_0;
  LOG(INFO) << "transpose_1: "<< transpose_1;
}

template <typename Dtype>
void MatrixMulLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  //----Move to Reshape
  //get bottom shape
  const vector<int> shape_0 = bottom[0]->shape();
  const vector<int> shape_1 = bottom[1]->shape();

  //we will assume that each blob is at most 3 dimensional
  CHECK_EQ(shape_0.size(),3) << "Blob 0" <<" must be 3D";
  CHECK_EQ(shape_1.size(),3) << "Blob 1" <<" must be 3D";
  
  //either the channels should match or they should be one
  if(shape_0[0] != shape_1[0]){
    if (shape_0[0] > shape_1[0])
      CHECK_EQ(shape_0[0] % shape_1[0], 0) << "Blob 0 has " << shape_0[0] << " channels, blob 1 has " << shape_1[0] << " channels.";
    else
      CHECK_EQ(shape_1[0] % shape_0[0], 0) << "Blob 0 has " << shape_0[0] << " channels, blob 1 has " << shape_1[0] << " channels.";
  }
  
  vector<int> shape(3);
  ch_0 = shape_0[0]; ch_1 = shape_1[0];
  maxch = ch_0 > ch_1 ? ch_0 : ch_1;
  shape[0] = maxch;
  int K1, K0, C0, C1;  
  if(transpose_0){
    C0 = shape_0[2]; K0 = shape_0[1];
  }
  else{
    C0 = shape_0[1]; K0 = shape_0[2];
  }

  if(transpose_1){
    C1 = shape_1[1]; K1 = shape_1[2];
  }
  else{
    C1 = shape_1[2]; K1 = shape_1[1];
  }
  CHECK_EQ(K0, K1) << "Matrix mult. rules";
  K = K1; M = C0; N = C1; 
  shape[1] = C0; shape[2] = C1;
  top[0]->Reshape(shape);
} 

template<typename Dtype>
void MatrixMulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int offset_data_1 = 0, offset_data_0 = 0;
  
  //the first bottom
  const Dtype* bottom_data_0 = bottom[0]->cpu_data();
  int count0 = bottom[0]->count();
  
  //the second bottom  
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  int count1 = bottom[1]->count();

  Dtype* top_data = top[0]->mutable_cpu_data();
  for(int i=0; i<maxch; i++){
    //do matrix multiplication
    if(transpose_0){
      if(transpose_1)
        caffe_cpu_gemm(CblasTrans, CblasTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
      else
        caffe_cpu_gemm(CblasTrans, CblasNoTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
    }
    else{
      if(transpose_1)
        caffe_cpu_gemm(CblasNoTrans, CblasTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
      else
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M, N, K, Dtype(1.0), bottom_data_0 + offset_data_0,
                      bottom_data_1 + offset_data_1, Dtype(0.0), top_data);
    }
    top_data+=M*N;
    
    offset_data_0 +=M*K; offset_data_0 = offset_data_0 % count0;
    offset_data_1 +=K*N; offset_data_1 = offset_data_1 % count1;
  }
}

template<typename Dtype>
void MatrixMulLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //all the bottoms 
  if(!propagate_down[0] && !propagate_down[1]){
    return;
  }
  Dtype *bottom_diff_0, *bottom_diff_1;
  const Dtype* top_diff = top[0]->cpu_diff();
  if(propagate_down[0]){
    bottom_diff_0 = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff_0);
  }
  if(propagate_down[1]){
    bottom_diff_1 = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0.0), bottom_diff_1);
  }

  //will also need the datas
  int offset_data_1 = 0, offset_data_0 = 0, offset_diff_0 = 0, offset_diff_1 = 0;
  
  const Dtype* bottom_data_0 = bottom[0]->cpu_data();
  int count0 = bottom[0]->count();
  
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  int count1 = bottom[1]->count();

  for(int i=0; i<maxch; i++){
    if(propagate_down[0]){
      if(transpose_0){
        if(transpose_1)
          caffe_cpu_gemm(CblasTrans, CblasTrans, K, M, N, Dtype(1.0),
                     bottom_data_1 + offset_data_1, top_diff, Dtype(1.0), bottom_diff_0 + offset_diff_0);
        else
          caffe_cpu_gemm(CblasNoTrans, CblasTrans, K, M, N, Dtype(1.0),
                     bottom_data_1 + offset_data_1, top_diff, Dtype(1.0), bottom_diff_0 + offset_diff_0);
      }
      else{
        if(transpose_1)
          caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M, K, N, Dtype(1.0), top_diff,
                     bottom_data_1 + offset_data_1, Dtype(1.0), bottom_diff_0 + offset_diff_0);
        else
          caffe_cpu_gemm(CblasNoTrans, CblasTrans, M, K, N, Dtype(1.0), top_diff,
                     bottom_data_1 + offset_data_1, Dtype(1.0), bottom_diff_0 + offset_diff_0);
      }
    }
    if(propagate_down[1]){
      if(transpose_0){
        if(transpose_1)
          caffe_cpu_gemm(CblasTrans, CblasTrans, N, K, M, Dtype(1.0), 
                        top_diff, bottom_data_0 + offset_data_0, Dtype(1.0), bottom_diff_1 + offset_diff_1);
        else
          caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K, N, M, Dtype(1.0), bottom_data_0 + offset_data_0,
                        top_diff, Dtype(1.0), bottom_diff_1 + offset_diff_1);
      }
      else{
        if(transpose_1)
          caffe_cpu_gemm(CblasTrans, CblasNoTrans, N, K, M, Dtype(1.0), 
                        top_diff, bottom_data_0 + offset_data_0, Dtype(1.0), bottom_diff_1 + offset_diff_1);
        else
          caffe_cpu_gemm(CblasTrans, CblasNoTrans, K, N, M, Dtype(1.0), bottom_data_0 + offset_data_0,
                        top_diff, Dtype(1.0), bottom_diff_1 + offset_diff_1);
      }
    }
    top_diff += M*N;
    offset_data_0 += M*K; offset_data_0 = offset_data_0 % count0;
    offset_diff_0 += M*K; offset_diff_0 = offset_diff_0 % count0;
    
    offset_data_1 += K*N; offset_data_1 = offset_data_1 % count1;
    offset_diff_1 += K*N; offset_diff_1 = offset_diff_1 % count1;
  }
}


#ifdef CPU_ONLY
STUB_GPU(MatrixMulLayer);
#endif

INSTANTIATE_CLASS(MatrixMulLayer);
REGISTER_LAYER_CLASS(MatrixMul);
}
