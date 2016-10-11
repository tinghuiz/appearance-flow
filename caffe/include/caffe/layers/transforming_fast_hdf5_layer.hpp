#ifndef CAFFE_TRANSFORMING_FAST_HDF5_LAYER_HPP_
#define CAFFE_TRANSFORMING_FAST_HDF5_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/fast_hdf5_layer.hpp"
#include "caffe/layers/transformation_layer.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Read blobs from disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class TransformingFastHDF5InputLayer : public Layer<Dtype>,
  protected InternalThread {
 public:
  explicit TransformingFastHDF5InputLayer(const LayerParameter& param);
  virtual ~TransformingFastHDF5InputLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "TransformingFastHDF5Input";
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinNumNumTopBlobs() const { return 1; }

 protected:
  shared_ptr<Layer<Dtype> > input_layer_, transformation_layer_;

  std::vector<Blob<Dtype>*> tmp_;
  virtual void InternalThreadEntry();
  struct Batch {
    std::vector<std::vector<Blob<Dtype>*> > blobs_;
  };
  virtual void load_batch(Batch* batch);
  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;
  Batch prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch*> prefetch_free_;
  BlockingQueue<Batch*> prefetch_full_;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace caffe

#endif  // CAFFE_TRANSFORMING_FAST_HDF5_LAYER_HPP_
