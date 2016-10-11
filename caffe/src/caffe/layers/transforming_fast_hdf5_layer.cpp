#include <boost/thread.hpp>
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/transforming_fast_hdf5_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype> TransformingFastHDF5InputLayer<Dtype>::
TransformingFastHDF5InputLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {
  // Set the BS to 1 before we create the layer
  LayerParameter p = param;
  p.mutable_fast_hdf5_input_param()->set_batch_size(1);
  input_layer_.reset(new FastHDF5InputLayer<Dtype>(p));
  transformation_layer_.reset(new TransformationLayer<Dtype>(p));
}

template<typename Dtype> TransformingFastHDF5InputLayer<Dtype>::
~TransformingFastHDF5InputLayer() {
  StopInternalThread();
  for (int i = 0; i < tmp_.size(); i++)
    delete tmp_[i];
  for (int i = 0; i < PREFETCH_COUNT; i++)
    for (int j = 0; j < prefetch_[i].blobs_.size(); j++)
      for (int k = 0; k < prefetch_[i].blobs_[j].size(); k++)
        delete prefetch_[i].blobs_[j][k];
}

template<typename Dtype> void TransformingFastHDF5InputLayer<Dtype>::
LayerSetUp(const vector<Blob<Dtype>*>& b, const vector<Blob<Dtype>*>& t) {
  int batch_size = this->layer_param_.fast_hdf5_input_param().batch_size();

  for (int i = 0; i < PREFETCH_COUNT; i++) {
    prefetch_[i].blobs_.resize(batch_size);
    for (int j = 0; j < batch_size; j++) {
      prefetch_[i].blobs_[j].resize(t.size());
      for (int k = 0; k < t.size(); k++)
        prefetch_[i].blobs_[j][k] = new Blob<Dtype>();
    }
    prefetch_free_.push(prefetch_ + i);
  }
  tmp_.resize(t.size());
  for (int i = 0; i < tmp_.size(); i++)
    tmp_[i] = new Blob<Dtype>();
  input_layer_->LayerSetUp(b, tmp_);
  transformation_layer_->LayerSetUp(tmp_, prefetch_[0].blobs_[0]);
  StartInternalThread();
}
template<typename Dtype> void TransformingFastHDF5InputLayer<Dtype>::
Reshape(const vector<Blob<Dtype>*>& b, const vector<Blob<Dtype>*>& t) {
  Batch* batch = prefetch_full_.peek();
  int batch_size = this->layer_param_.fast_hdf5_input_param().batch_size();
  for (int n = 0; n < t.size(); n++) {
    vector<int> s = batch->blobs_[0][n]->shape();
    CHECK_EQ(s[0], 1) << "Transformation output should have batch size 1";
    s[0] = batch_size;
    t[n]->Reshape(s);
  }
}
template<typename Dtype> void TransformingFastHDF5InputLayer<Dtype>::
Forward_cpu(const vector<Blob<Dtype>*>& b, const vector<Blob<Dtype>*>& t) {
  Batch* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  int batch_size = this->layer_param_.fast_hdf5_input_param().batch_size();
  for (int i = 0; i < batch_size; i++)
    for (int n = 0; n < t.size(); n++)
      caffe_copy(batch->blobs_[i][n]->count(), batch->blobs_[i][n]->cpu_data(),
                 t[n]->mutable_cpu_data() + t[n]->offset(i));
  prefetch_free_.push(batch);
}
template<typename Dtype> void TransformingFastHDF5InputLayer<Dtype>::
Forward_gpu(const vector<Blob<Dtype>*>& b, const vector<Blob<Dtype>*>& t) {
  Batch* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  int batch_size = this->layer_param_.fast_hdf5_input_param().batch_size();
  for (int i = 0; i < batch_size; i++)
    for (int n = 0; n < t.size(); n++)
      caffe_copy(batch->blobs_[i][n]->count(), batch->blobs_[i][n]->gpu_data(),
                 t[n]->mutable_gpu_data() + t[n]->offset(i));
  prefetch_free_.push(batch);
}
template <typename Dtype>
void TransformingFastHDF5InputLayer<Dtype>::InternalThreadEntry() {
  try {
    while (!must_stop()) {
      Batch* batch = prefetch_free_.pop();
      load_batch(batch);
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}
template <typename Dtype>
void TransformingFastHDF5InputLayer<Dtype>::load_batch(Batch* batch) {
  int batch_size = this->layer_param_.fast_hdf5_input_param().batch_size();
  for (int i = 0; i < batch_size; i++) {
    input_layer_->Reshape(vector<Blob<Dtype>*>(), tmp_);
    input_layer_->Forward(vector<Blob<Dtype>*>(), tmp_);
    transformation_layer_->Reshape(tmp_, batch->blobs_[i]);
    transformation_layer_->Forward(tmp_, batch->blobs_[i]);
    for (int n = 0; n < batch->blobs_[i].size(); n++)
      CHECK(batch->blobs_[0][n]->shape() == batch->blobs_[i][n]->shape()) <<
        "Transformation outputs different shape";
  }
}

INSTANTIATE_CLASS(TransformingFastHDF5InputLayer);
REGISTER_LAYER_CLASS(TransformingFastHDF5Input);

}  // namespace caffe
