#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <vector>

#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief Color space conversion layer
 *
 *   This layer converts between various color spaces such as RGB, BGR, Lab,
 *   Luv, XYZ, Gray. RGB and BGR use the sRGB color space and conversion.
 *   To keep the caffe convention RGB is scaled 0..255, Gray is scales 0..255,
 *   and the other color spaces are scaled according to their CIE definitions.
 */
template <typename Dtype>
class ColorConvLayer : public Layer<Dtype> {
 public:
  explicit ColorConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ColorConv"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  ColorConvParameter::ColorSpace input_space_, output_space_;
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
