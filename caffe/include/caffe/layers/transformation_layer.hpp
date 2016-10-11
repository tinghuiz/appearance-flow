#ifndef CAFFE_TRANSFORM_LAYER_HPP_
#define CAFFE_TRANSFORM_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief Applies data transformations to a number of bottom blobs.
 *        The number of bottom and top blobs must match exactly and the shape of
 *        all bottom blobs needs to be the same. It is also expected that the
 *        bottom blobs are images N x C x H x W
 */
template <typename Dtype>
class TransformationLayer : public Layer<Dtype> {
 public:
  struct Affine2D {
    Dtype a00_, a01_, a10_, a11_, t0_, t1_;
    Affine2D(Dtype a00, Dtype a01, Dtype a10, Dtype a11, Dtype t0, Dtype t1):
      a00_(a00), a01_(a01), a10_(a10), a11_(a11), t0_(t0), t1_(t1) {}
    Affine2D operator*(const Affine2D & o) {
      return Affine2D(a00_*o.a00_+a01_*o.a10_, a00_*o.a01_+a01_*o.a11_,
                      a10_*o.a00_+a11_*o.a10_, a10_*o.a01_+a11_*o.a11_,
                      a00_*o.t0_+a01_*o.t1_+t0_, a10_*o.t0_+a11_*o.t1_+t1_);
    }

    Dtype x(Dtype x, Dtype y) { return a00_*x + a01_*y + t0_; }
    Dtype y(Dtype x, Dtype y) { return a10_*x + a11_*y + t1_; }
  };

  explicit TransformationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Transform"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  int crop_size_;
  std::vector<shared_ptr<Blob<Dtype> > > mean_value_;
  bool mirror_, rotate_, synchronized_;
  Dtype scale_, min_scale_, max_scale_;
  std::vector<Affine2D> generate(int N, int W, int H, int W_out, int H_out);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
};

}  // namespace caffe

#endif  // CAFFE_TRANSFORM_LAYER_HPP_
