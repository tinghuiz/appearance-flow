#ifndef CAFFE_MATRIX_MUL_LAYER_HPP_
#define CAFFE_MATRIX_MUL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


/*
 * A Matrix multiplication layer
 */

template <typename Dtype>
class MatrixMulLayer : public Layer<Dtype> {
  public:
  explicit MatrixMulLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MatrixMul"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int ch_1, ch_0, maxch, M, N, K ;
  bool streams_need_init_, use_streams_;
  cudaStream_t *stream_, *default_stream;
  int max_streams_;
  bool transpose_0, transpose_1;
};

}  // namespace caffe

#endif  // CAFFE_MATRIX_MUL_LAYER_HPP_
