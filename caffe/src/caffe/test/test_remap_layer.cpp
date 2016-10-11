#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RemapLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  RemapLayerTest()
    : blob_bottom_0_(new Blob<Dtype>(2, 5, 6, 5)),
      blob_bottom_1_(new Blob<Dtype>(2, 2, 3, 7)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-5.);
    filler_param.set_max(10.);
    shared_ptr<UniformFiller<Dtype> > filler;
    filler.reset(new UniformFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0_);
    filler.reset(new UniformFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~RemapLayerTest() { delete blob_bottom_0_; delete blob_bottom_1_; delete blob_top_; }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RemapLayerTest, TestDtypesAndDevices);

TYPED_TEST(RemapLayerTest, TestRemapGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RemapLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, 
    this->blob_top_vec_);
}

// TYPED_TEST(RemapLayerTest, TestReLU) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   ReLULayer<Dtype> layer(layer_param);
//   // FIXME: setup two bottoms, one top?
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer.Forward(...);


// }

}  // namespace caffe
