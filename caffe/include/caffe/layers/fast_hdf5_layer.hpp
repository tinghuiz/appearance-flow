#ifndef CAFFE_FAST_HDF5_LAYER_HPP_
#define CAFFE_FAST_HDF5_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "hdf5.h"

namespace caffe {

/**
 * @brief Read a single blob from an HDF5 file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class StaticHDF5InputLayer : public Layer<Dtype> {
 public:
  explicit StaticHDF5InputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StaticHDF5Input"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinNumNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  std::vector<shared_ptr<Blob<Dtype> > > blobs_;
  int batch_size_;
};

/**
 * @brief Read blobs from a single HDF5 file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FastHDF5InputLayer : public Layer<Dtype>, protected InternalThread {
 public:
  explicit FastHDF5InputLayer(const LayerParameter& param)
      : Layer<Dtype>(param), file_id_(-1) {}
  virtual ~FastHDF5InputLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FastHDF5Input"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinNumNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  virtual void InternalThreadEntry();

  struct Batch {
    shared_ptr<SyncedMemory> memory_;
    std::vector<std::vector<int> > shape_;
    std::vector<int> type_;
  };
  virtual void load_batch(Batch* batch);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;
  Batch prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch*> prefetch_free_;
  BlockingQueue<Batch*> prefetch_full_;

  int batch_size_, file_it_, n_images_;  /* Information about the read status */
  hid_t file_id_;
  std::vector<hid_t> groups_;
  shared_ptr<SyncedMemory> gpu_tmp_;
};

/**
 * @brief Write blobs to disk to a single HDF5 file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FastHDF5OutputLayer : public Layer<Dtype>/*, protected InternalThread*/ {
 public:
  explicit FastHDF5OutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param), file_id_(-1) {}
  virtual ~FastHDF5OutputLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FastHDF5Output"; }
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }
  static hid_t half_t();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void saveData(const vector<Blob<Dtype>*>& bottom);

  int it_, type_;
  hid_t file_id_;
  std::vector<hid_t> groups_;
  std::vector<size_t> offset_;
  shared_ptr<SyncedMemory> gpu_tmp_, cpu_tmp_;
};

}  // namespace caffe

#endif  // CAFFE_FAST_HDF5_LAYER_HPP_
