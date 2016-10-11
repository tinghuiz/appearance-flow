#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename T>
static void setupLerp(std::vector<int> *id0, std::vector<int> *id1,
                      std::vector<T> *w, int N, int NN, T s, T crop) {
  id0->resize(N);
  id1->resize(N);
  w->resize(N);
  for (int i = 0; i < N; i++) {
    T x = i / s + crop;
    if (x < 0) x = 0;
    if (x > NN-1) x = NN-1;
    (*id0)[i] = x;
    (*id1)[i] = std::min((*id0)[i]+1, NN-1);
    (*w)[i] = (*id1)[i]-x;
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> S = bottom[0]->shape();
  const int D = S.size();
  CHECK_GE(D, 2) << "At least 2 dimensions expected";

  ResizeParameter param = this->layer_param_.resize_param();

  // Get the crop
  crop_x_ = crop_y_ = param.crop();
  if (param.has_crop_x()) crop_x_ = param.crop_x();
  if (param.has_crop_y()) crop_y_ = param.crop_y();

  // Compute the scaling factor
  scale_x_ = scale_y_ = 0.f;
  if (param.has_scale() && (param.has_scale_x() || param.has_scale_y()))
    LOG(WARNING) << "Cannot specify scale and scalex or scaly";
  if (param.has_scale()) scale_x_ = scale_y_ = param.scale();
  if (param.has_scale_x()) scale_x_ = param.scale_x();
  if (param.has_scale_y()) scale_y_ = param.scale_y();

  // Try to compute the scale from the shape
  if (param.has_width()) {
    CHECK_LE(scale_x_, 0.f) << "Multiple parameters defining scale";
    if (param.width() == 0) scale_x_ = 1.f;
    else
      scale_x_ = (param.width()-1.f) / (S[D-1]-2*crop_x_-1);
  }
  if (param.has_height()) {
    CHECK_LE(scale_y_, 0.f) << "Multiple parameters defining scale";
    if (param.height() == 0) scale_y_ = 1.f;
    else
      scale_y_ = (param.height()-1.f) / (S[D-2]-2*crop_y_-1);
  }
  const int W = S[D-1], H = S[D-2];
  S[D-1] = scale_x_ * (W-2*crop_x_-1)+1 + 0.5/*rounding*/;
  S[D-2] = scale_y_ * (H-2*crop_y_-1)+1 + 0.5/*rounding*/;
  if (bottom.size() > 1) {
    std::vector<int> s = bottom[1]->shape();
    CHECK_GE(s.size(), 2) << "bottom[1]: At least two dimensions required";
    if (scale_x_ == 0) scale_x_ = (s[s.size()-1]-1.f) / (W-2*crop_x_-1);
    if (scale_y_ == 0) scale_y_ = (s[s.size()-2]-1.f) / (H-2*crop_y_-1);
    S[D-1] = s[D-1];
    S[D-2] = s[D-2];
  }
  top[0]->Reshape(S);

  // Allocate the temp memory
  if (W > S[D-1]) S[D-1] = W;
  if (H > S[D-2]) S[D-2] = H;
  tmp_.Reshape(S);
}
template <typename T>
static void tentx_cpu(const T* in, int W, int H, int D, T* out, T R) {
  int iR = R;
  T norm = ((2*iR+1)*R - iR*(iR+1));
  // I know there is linear time algorithms for this, but it's annoying to code
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < D; i++)
    for (int y = 0, k = i*W*H; y < H; y++ )
      for (int x = 0; x < W; x++, k++) {
        T s = R*in[k];
        for (int r = 1; r < R; r++)
          s += (R-r)*(in[x-r >= 0 ? k-r : k-x] + in[x+r < W ? k+r : k-x+W-1]);
        out[k] = s / norm;
      }
}
template <typename T>
static void tenty_cpu(const T* in, int W, int H, int D, T* out, T R) {
  int iR = R;
  T norm = ((2*iR+1)*R - iR*(iR+1));
  // I know there is linear time algorithms for this, but it's annoying to code
  std::vector<T> s(W);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < D; i++)
    for (int y = 0; y < H; y++) {
      std::fill(s.begin(), s.end(), T(0));
      for (int x = 0, k = (i*H+y)*W; x < W; x++, k++) {
        s[x] = R*in[k];
        for (int r = 1; r < R; r++ )
          s[x] += (R-r)*(in[y-r >= 0 ? k-r*W : k-y*W] +
                         in[y+r < H ? k+r*W : k+(H-1-y)*W]);
      }
      for (int x = 0, k = (i*H+y)*W; x < W; x++, k++ )
        out[k] = s[x] / norm;
    }
}
template <typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const std::vector<int> & S = bottom[0]->shape(), &tS = top[0]->shape();
  const int W = S[S.size()-1], H = S[S.size()-2];
  const int D = bottom[0]->count() / (W*H);
  const int tW = tS[tS.size()-1], tH = tS[tS.size()-2];
  std::vector<int> id0, id1;
  std::vector<Dtype> w;

  // Apply the blur
  const Dtype * p_bot = bottom[0]->cpu_data();
  Dtype * p_tmp0 = tmp_.mutable_cpu_data();
  Dtype * p_tmp1 = tmp_.mutable_cpu_diff();
  Dtype * p_top = top[0]->mutable_cpu_data();

  // Resize in x
  if (scale_x_ < 1) {
    // If the sampling was perfect (no interpolation later), then a tent filter
    // with radius 1/scale will distribute all bottom values between the top
    // samples
    tentx_cpu(p_bot, W, H, D, p_tmp0, Dtype(1.)/std::max(scale_x_, Dtype(.01)));
    p_bot = p_tmp0;
  }
  setupLerp<Dtype>(&id0, &id1, &w, tW, W, scale_x_, crop_x_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < D; i++)
    for (int y = 0, k = i*tW*H; y < H; y++ )
      for (int x = 0; x < tW; x++, k++ )
        p_tmp1[k] =    w[x] *p_bot[i*W*H + y*W + id0[x]] +
                    (1-w[x])*p_bot[i*W*H + y*W + id1[x]];

  // Resize in y
  if (scale_y_ < 1) {
    tenty_cpu(p_tmp1, tW, H, D, p_tmp0,
              Dtype(1.)/std::max(scale_y_, Dtype(.01)));
    std::swap(p_tmp1, p_tmp0);
  }
  setupLerp<Dtype>(&id0, &id1, &w, tH, H, scale_y_, crop_y_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < D; i++)
    for (int y = 0, k = i*tW*tH; y < tH; y++ )
      for (int x = 0; x < tW; x++, k++ )
        p_top[k] =    w[y] *p_tmp1[i*tW*H + id0[y]*tW + x] +
                   (1-w[y])*p_tmp1[i*tW*H + id1[y]*tW + x];
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const std::vector<int> & S = bottom[0]->shape(), &tS = top[0]->shape();
  const int W = S[S.size()-1], H = S[S.size()-2];
  const int D = bottom[0]->count() / (W*H);
  const int tW = tS[tS.size()-1], tH = tS[tS.size()-2];
  std::vector<int> id0, id1;
  std::vector<Dtype> w;

  Dtype * p_bot = bottom[0]->mutable_cpu_diff();
  Dtype * p_tmp0 = tmp_.mutable_cpu_data();
  Dtype * p_tmp1 = tmp_.mutable_cpu_diff();
  const Dtype * p_top = top[0]->cpu_diff();

  // Resize in y
  caffe_set(D*tW*H, Dtype(0), p_tmp1);
  setupLerp<Dtype>(&id0, &id1, &w, tH, H, scale_y_, crop_y_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < D; i++)
    for (int y = 0, k = i*tW*tH; y < tH; y++)
      for (int x = 0; x < tW; x++, k++) {
        p_tmp1[i*tW*H + id0[y]*tW + x] +=    w[y] *p_top[k];
        p_tmp1[i*tW*H + id1[y]*tW + x] += (1-w[y])*p_top[k];
      }
  if (scale_y_ < 1) {
    tenty_cpu(p_tmp1, tW, H, D, p_tmp0, Dtype(1.)/scale_y_);
    std::swap(p_tmp1, p_tmp0);
  }

  // Resize in x
  caffe_set(bottom[0]->count(), Dtype(0), p_bot);
  setupLerp<Dtype>(&id0, &id1, &w, tW, W, scale_x_, crop_x_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < D; i++)
    for (int y = 0, k = i*tW*H; y < H; y++)
      for (int x = 0; x < tW; x++, k++) {
        p_bot[i*W*H + y*W + id0[x]] +=    w[x] *p_tmp1[k];
        p_bot[i*W*H + y*W + id1[x]] += (1-w[x])*p_tmp1[k];
      }
  if (scale_x_ < 1) {
    caffe_copy(W*H*D, p_bot, p_tmp0);
    tentx_cpu(p_tmp0, W, H, D, p_bot, Dtype(1.)/scale_x_);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ResizeLayer);
#endif

INSTANTIATE_CLASS(ResizeLayer);
REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe
