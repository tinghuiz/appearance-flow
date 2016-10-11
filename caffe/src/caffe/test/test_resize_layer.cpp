#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ResizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ResizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 10, 10)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ResizeLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ResizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ResizeLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  // Try a simple upsampling
  resize_param->set_scale(2);
  shared_ptr<Layer<Dtype> > layer( new ResizeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 11);
  EXPECT_EQ(this->blob_top_->width(), 7);

  // Try downsampling again
  resize_param->set_scale(0.5);
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_top_);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_2_);

  layer.reset(new ResizeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_2_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_2_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_2_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_2_->width(), this->blob_bottom_->width());

  // Try matching the bottom2 shape
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  resize_param->clear_scale();
  layer.reset(new ResizeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_2_->num(), this->blob_bottom_2_->num());
  EXPECT_EQ(this->blob_top_2_->channels(), this->blob_bottom_2_->channels());
  EXPECT_EQ(this->blob_top_2_->height(), this->blob_bottom_2_->height());
  EXPECT_EQ(this->blob_top_2_->width(), this->blob_bottom_2_->width());
}

TYPED_TEST(ResizeLayerTest, TestIdentity) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  resize_param->set_scale(1);
  shared_ptr<Layer<Dtype> > layer( new ResizeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* bot_data = this->blob_bottom_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i)
    EXPECT_NEAR(top_data[i], bot_data[i], 1e-4);
}

TYPED_TEST(ResizeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  for (float f = 0.5; f < 2.2; f += 0.5) {
    resize_param->set_scale(f);
    ResizeLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

}  // namespace caffe
