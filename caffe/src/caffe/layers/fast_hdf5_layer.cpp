#include <boost/thread.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <H5Zpublic.h>  // NOLINT[build/include_alpha]
#ifdef USE_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#ifdef LZF_LIB
extern "C" {
#include <dlfcn.h>
}
#endif

#include "caffe/layers/fast_hdf5_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

static hid_t half_t() {
  static hid_t half = -1;
  if (half == -1) {
    half = H5Tcopy(H5T_IEEE_F32LE);
    H5Tset_fields(half, 15, 10, 5, 0, 10);
    H5Tset_size(half, 2);
    H5Tset_ebias(half, 15);
    H5Tlock(half);
  }
  return half;
}
static int count(const std::vector<int> & s) {
  int r = 1;
  for (int i = 0; i < s.size(); i++)
    r *= s[i];
  return r;
}
static int data_size(int type) {
  if (type == UINT8) return 1;
  if (type == FLOAT16) return 2;
  if (type == FLOAT32) return 4;
  LOG(FATAL) << "Unknown data type";
  return 0;
}
static hid_t to_h5t(int type) {
  if (type == UINT8) return H5T_NATIVE_UCHAR;
  if (type == FLOAT16) return half_t();
  if (type == FLOAT32) return H5T_NATIVE_FLOAT;
  LOG(FATAL) << "Unknown data type";
  return 0;
}
template <typename Dtype>
hid_t FastHDF5OutputLayer<Dtype>::half_t() {
  return caffe::half_t();
}
typedef uint16_t half;
template<typename T1, typename T2>
std::vector<T1> convertVec(const std::vector<T2> &v) {
  std::vector<T1> r(v.size());
  std::copy(v.begin(), v.end(), r.begin());
  return r;
}
template <typename Dtype>
FastHDF5OutputLayer<Dtype>::~FastHDF5OutputLayer() {
  for (int i = 0; i < groups_.size(); i++)
    H5Gclose(groups_[i]);
  if (file_id_ >= 0)
    H5Fclose(file_id_);
}
static std::string unSplit(const std::string & s) {
  if (s.find("_split_") == std::string::npos) return s;
  return s.substr(0, s.find("_"));
}
template <typename Dtype>
void FastHDF5OutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  FastHDF5OutputParameter param = this->layer_param_.fast_hdf5_output_param();
  // Open the HDF5 file
  std::string source = param.file_name();
  file_id_ = H5Fcreate(source.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  // Create a group for each bottom
  groups_.clear();
  for (int i = 0; i < bottom.size(); i++) {
    std::string n = "/"+unSplit(this->layer_param_.bottom(i));
    if (i < param.group_name_size())
      n = param.group_name(i);
    hid_t g = H5Gcreate(file_id_, n.c_str(), H5P_DEFAULT, H5P_DEFAULT,
                        H5P_DEFAULT);
    groups_.push_back(g);
  }
  // Compute the HDF5 data type
  type_ = param.storage();
  // Reset the iterations to 0
  it_ = 0;
}

template <typename Dtype>
void FastHDF5OutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  size_t datasize = data_size(type_), max_size = 0;
  offset_.clear();
  for (int i = 0; i < bottom.size(); i++) {
    offset_.push_back(max_size*datasize);
    max_size += bottom[i]->count();
  }

  cpu_tmp_.reset(new SyncedMemory(max_size*datasize));
  gpu_tmp_.reset(new SyncedMemory(max_size*datasize));
}
template<typename T1, typename T2>
static void convertCPU(const T1 * from, T2 * to, int N) {
  std::copy(from, from+N, to);
}
union FP32 {
  uint u;
  float f;
  struct {
    uint Mantissa : 23;
    uint Exponent : 8;
    uint Sign : 1;
  };
};
template<typename T1>
static void convertCPU(const T1 * from, half * to, int N) {
  FP32 magic = {15 << 23};
  const uint f32infty = 255 << 23, f16infty = 31 << 23;
  const uint sign_mask = 0x80000000u, round_mask = ~0xfffu;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++) {
    FP32 f32; f32.f = from[i];
    half f16 = 0;
    uint sign = f32.u & sign_mask;
    f32.u ^= sign;
    if (f32.u >= f32infty) {  // Inf or NaN (all exponent bits set)
      f16 = (f32.u > f32infty) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
    } else {  // (De)normalized number or zero
      f32.u &= round_mask;
      f32.f *= magic.f;
      f32.u -= round_mask;
      // Clamp to signed infinity if overflowed
      if (f32.u > f16infty) f32.u = f16infty;
      f16 = f32.u >> 13;  // Take the bits!
    }
    f16 |= sign >> 16;
    to[i] = f16;
  }
}
template<typename T2>
static void convertCPU(const half * from, T2 * to, int N) {
  static const FP32 magic = { 113 << 23 };
  static const uint shifted_exp = 0x7c00 << 13;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; i++) {
    half f16 = from[i];
    FP32 f32;
    f32.u = (f16 & 0x7fff) << 13;     // exponent/mantissa bits
    uint exp = shifted_exp & f32.u;   // just the exponent
    f32.u += (127 - 15) << 23;        // exponent adjust
    // handle exponent special cases
    if (exp == shifted_exp) {       // Inf/NaN?
      f32.u += (128 - 16) << 23;    // extra exp adjust
    } else if (exp == 0) {          // Zero/Denormal?
      f32.u += 1 << 23;             // extra exp adjust
      f32.f -= magic.f;             // renormalize
    }
    f32.u |= (f16 & 0x8000) << 16;  // sign bit
    to[i] = f32.f;
  }
}
template <typename Dtype>
void FastHDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  // Convert the data to the cpu_buffer
  char * mutable_cpu_tmp = static_cast<char*>(cpu_tmp_->mutable_cpu_data());
  for (int i = 0; i < bottom.size(); i++) {
    char * p_cpu = mutable_cpu_tmp+offset_[i];
    if (type_ == UINT8)
      convertCPU(bottom[i]->cpu_data(), reinterpret_cast<uint8_t*>(p_cpu),
                 bottom[i]->count());
    else if (type_ == FLOAT16)
      convertCPU(bottom[i]->cpu_data(), reinterpret_cast<half*>(p_cpu),
                 bottom[i]->count());
    else if (type_ == FLOAT32)
      convertCPU(bottom[i]->cpu_data(), reinterpret_cast<float*>(p_cpu),
                 bottom[i]->count());
    else
      LOG(FATAL) << "Unknown data type";
  }
  saveData(bottom);
}
template <typename Dtype>
void FastHDF5OutputLayer<Dtype>::saveData(const vector<Blob<Dtype>*>& bottom) {
  // Write the HDF5 file
  const char * cpu_tmp = static_cast<const char*>(cpu_tmp_->cpu_data());
  int batch_size = 0;
  for (int i = 0; i < bottom.size(); i++) {
    const char * p_cpu = cpu_tmp+offset_[i];
    std::vector<hsize_t> D = convertVec<hsize_t>(bottom[i]->shape());
    if (!batch_size)
      batch_size = D[0];
    CHECK_EQ(batch_size, D[0]) << "Only one batch size supported";
    D.erase(D.begin());
    for (int n = 0, it = it_; n < batch_size; n++, it++) {
      char it_name[64] = {};
      snprintf(it_name, sizeof(it_name), "%d", it);
      herr_t status = H5LTmake_dataset(groups_[i], it_name, D.size(), D.data(),
                                       to_h5t(type_), p_cpu);
      CHECK_GE(status, 0) << "Failed to make dataset " << it_name;
    }
  }
  H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
  it_ += batch_size;
}
#ifdef CPU_ONLY
STUB_GPU_FORWARD(FastHDF5OutputLayer);
#endif

template <typename Dtype>
FastHDF5InputLayer<Dtype>::~FastHDF5InputLayer() {
  StopInternalThread();
  for (int i = 0; i < groups_.size(); i++)
    H5Gclose(groups_[i]);
  if (file_id_ >= 0)
    H5Fclose(file_id_);
}
static std::vector<int> hdf5_info(hid_t f, const char * n, int * t) {
  int ndims = 0;
  H5LTget_dataset_ndims(f, n, &ndims);
  CHECK_GT(ndims, 0) << "Dataset need at least one dimension";
  std::vector<hsize_t> dims(ndims);
  H5T_class_t cls;
  size_t sz;
  H5LTget_dataset_info(f, n, dims.data(), &cls, &sz);
  if (t) {
    if (cls == H5T_INTEGER && sz == 1) *t = UINT8;
    else if (cls == H5T_FLOAT && sz == 4) *t = FLOAT32;
    else if (cls == H5T_FLOAT && sz == 2) *t = FLOAT16;
    else
      LOG(FATAL) << "Unknown hdf5 data type";
  }
  std::vector<int> idims(dims.size());
  std::copy(dims.begin(), dims.end(), idims.begin());
  return idims;
}
static bool hdf5_exists(hid_t f, const char * name) {
  return H5Lexists(f, name, H5P_DEFAULT);
}



template <typename Dtype>
void StaticHDF5InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  FastHDF5InputParameter param = this->layer_param_.fast_hdf5_input_param();
  batch_size_ = param.batch_size();

  std::string source = param.source();
  hid_t fid = H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(fid, 0) << "File '" << source << "' not found!";

  blobs_.clear();
  // Load the different groups
  for (int i = 0; i < top.size(); i++) {
    std::string n = this->layer_param_.top(i);
    if (i < param.group_name_size())
      n = param.group_name(i);
    CHECK(hdf5_exists(fid, n.c_str())) << "Dataset '" << n << "' not found!";

    int type = 0;
    std::vector<int> shape = hdf5_info(fid, n.c_str(), &type);
    std::vector<char> data(count(shape) * data_size(type));

    // Read the data
    herr_t status = H5LTread_dataset(fid, n.c_str(), to_h5t(type), &data[0]);
    CHECK_GE(status, 0) << "No dataset named '" << n << "' found!";

    shared_ptr<Blob<Dtype> > blob( new Blob<Dtype>(shape) );
    if (type == UINT8)
      convertCPU(reinterpret_cast<const uint8_t*>(&data[0]),
                 blob->mutable_cpu_data(), blob->count());
    else if (type == FLOAT16)
      convertCPU(reinterpret_cast<const half*>(&data[0]),
                 blob->mutable_cpu_data(), blob->count());
    else if (type == FLOAT32)
      convertCPU(reinterpret_cast<const float*>(&data[0]),
                 blob->mutable_cpu_data(), blob->count());
    else
      LOG(FATAL) << "Unkown converter";
    blobs_.push_back(blob);
  }
  H5Fclose(fid);
}
template <typename Dtype>
void StaticHDF5InputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); i++) {
    std::vector<int> S = blobs_[i]->shape();
    S.insert(S.begin(), batch_size_);
    top[i]->Reshape(S);
  }
}
template <typename Dtype>
void StaticHDF5InputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>&,
                                              const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); i++) {
    for (int j = 0; j < batch_size_; j++) {
      caffe_copy(blobs_[i]->count(), blobs_[i]->cpu_data(),
                 top[i]->mutable_cpu_data() + top[i]->offset(j));
    }
  }
}
template <typename Dtype>
void StaticHDF5InputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& ,
                                              const vector<Blob<Dtype>*>& top) {
#ifdef CPU_ONLY
  NOT_IMPLMENTED;
#else
  for (int i = 0; i < top.size(); i++) {
    for (int j = 0; j < batch_size_; j++) {
      caffe_copy(blobs_[i]->count(), blobs_[i]->gpu_data(),
                 top[i]->mutable_gpu_data() + top[i]->offset(j));
    }
  }
#endif
}

template <typename Dtype>
void FastHDF5InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  FastHDF5InputParameter param = this->layer_param_.fast_hdf5_input_param();
  std::string source = param.source();
  file_id_ = H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Load the different groups
  groups_.clear();
  for (int i = 0; i < top.size(); i++) {
    std::string n = "/"+this->layer_param_.top(i);
    if (i < param.group_name_size())
      n = param.group_name(i);
    hid_t g = H5Gopen(file_id_, n.c_str(), H5P_DEFAULT);
    CHECK_GE(g, 0) << "Group '" << n << "' not found!";
    groups_.push_back(g);
  }

  // Set all variables required here!!!
  batch_size_ = param.batch_size();
  file_it_ = 0;
  // Find out how many images we have
  n_images_ = 0;
  if (groups_.size() > 0) {
    for (n_images_ = 1; ; n_images_*=2) {
      char buf[64];
      snprintf(buf, sizeof(buf), "%d", n_images_);
      if (!hdf5_exists(groups_[0], buf)) break;
    }
    int a = n_images_ / 2, b = n_images_;
    while (a < b) {
      n_images_ = (a+b) / 2;
      char buf[64];
      snprintf(buf, sizeof(buf), "%d", n_images_);
      if (hdf5_exists(groups_[0], buf))
        a = n_images_+1;
      else
        b = n_images_;
    }
    n_images_ = a;
  }
  // Assign all the prefetch data to the free nodes
  for (int i = 0; i < PREFETCH_COUNT; i++)
    prefetch_free_.push(prefetch_ + i);
  StartInternalThread();
}
template <typename Dtype>
void FastHDF5InputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  Batch* batch = prefetch_full_.peek();
  int max_size = 0;
  for (int i = 0; i < top.size(); i++) {
    std::vector<int> S = batch->shape_[i];
    S.insert(S.begin(), batch_size_);
    top[i]->Reshape(S);
    max_size = std::max(max_size, count(S) * data_size(batch->type_[i]));
  }
  if (!gpu_tmp_ || gpu_tmp_->size() < max_size)
    gpu_tmp_.reset(new SyncedMemory(max_size));
}
template <typename Dtype>
void FastHDF5InputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  Batch* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  const char * data = reinterpret_cast<const char*>(batch->memory_->cpu_data());
  size_t o = 0;
  for (int i = 0; i < top.size(); i++) {
    int type = batch->type_[i];
    size_t n = top[i]->count() * data_size(type);
    if (type == UINT8)
      convertCPU(reinterpret_cast<const uint8_t*>(data+o),
                 top[i]->mutable_cpu_data(), top[i]->count());
    else if (type == FLOAT16)
      convertCPU(reinterpret_cast<const half*>(data+o),
                 top[i]->mutable_cpu_data(), top[i]->count());
    else if (type == FLOAT32)
      convertCPU(reinterpret_cast<const float*>(data+o),
                 top[i]->mutable_cpu_data(), top[i]->count());
    else
      LOG(FATAL) << "Unkown converter";
    o += n;
  }
  prefetch_free_.push(batch);
}
template <typename Dtype>
void FastHDF5InputLayer<Dtype>::InternalThreadEntry() {
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
template <typename Dtype> void FastHDF5InputLayer<Dtype>::load_batch(
    FastHDF5InputLayer<Dtype>::Batch * b) {
  b->type_.resize(groups_.size());
  b->shape_.resize(groups_.size());

  // Read the group sizes and data types
  std::vector<size_t> dtsize_(groups_.size(), 0), offset_(groups_.size()+1, 0);
  for (int g = 0; g < groups_.size(); g++) {
    char it_name[64] = {};
    snprintf(it_name, sizeof(it_name), "%d", file_it_);

    // Get the hdf5 data info
    b->shape_[g] = hdf5_info(groups_[g], it_name, &b->type_[g]);
    dtsize_[g] = count(b->shape_[g]) * data_size(b->type_[g]);
    offset_[g+1] = offset_[g] + batch_size_ * dtsize_[g];
  }
  if (!b->memory_ || b->memory_->size() < offset_.back())
    b->memory_.reset(new SyncedMemory(offset_.back()));
  // Read as many blocks as needed
  char * cpu_mem = static_cast<char*>(b->memory_->mutable_cpu_data());
  for (int g = 0; g < groups_.size(); g++) {
    for (int i = 0, f = file_it_; i < batch_size_; i++, f++) {
      while (f >= n_images_) f -= n_images_;
      char it_name[64] = {};
      snprintf(it_name, sizeof(it_name), "%d", f);

      // Get the hdf5 data info
      int dt;
      std::vector<int> shape = hdf5_info(groups_[g], it_name, &dt);
      CHECK(shape == b->shape_[g]) << "Shape cannot change within mini batch";
      CHECK(dt == b->type_[g]) << "Data type cannot change within mini batch";

      // Read the data
      char *p_cpu = cpu_mem + offset_[g] + i * dtsize_[g];
      herr_t status = H5LTread_dataset(groups_[g], it_name, to_h5t(dt), p_cpu);
      CHECK_GE(status, 0) << "Failed to read dataset " << it_name;
    }
  }
  file_it_ += batch_size_;
  while (file_it_ >= n_images_) file_it_ -= n_images_;
}
#ifdef CPU_ONLY
STUB_GPU_FORWARD(FastHDF5InputLayer);
#endif


INSTANTIATE_CLASS(StaticHDF5InputLayer);
INSTANTIATE_CLASS(FastHDF5OutputLayer);
INSTANTIATE_CLASS(FastHDF5InputLayer);
REGISTER_LAYER_CLASS(StaticHDF5Input);
REGISTER_LAYER_CLASS(FastHDF5Output);
REGISTER_LAYER_CLASS(FastHDF5Input);

}  // namespace caffe
