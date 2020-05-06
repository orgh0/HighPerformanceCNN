#include <Container.hpp>
#include <utils.hpp>

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cmath>

Container::Container(const std::vector<int> &_shape) : shape(_shape) {
  int size = 1;
  for (auto it : _shape) {
    size *= it;
  }

  this->data.resize(size);
}

Container::Container(const std::vector<int> &_shape, float value) : shape(_shape) {
  int size = 1;
  for (auto it : _shape) {
    size *= it;
  }

  this->data.resize(size, value);
}

Container::Container(const std::vector<int> &_shape,
                 const std::vector<float> &_data)
    : shape(_shape), data(_data.begin(), _data.end()) {
  this->is_size_consistent();
}

Container::Container(const Container &other) { *this = other; }

Container &Container::operator=(const Container &other) {
  if (this != &other) {
    this->shape = other.shape;
    this->data = other.data;
  }

  return *this;
}

Container::Container(Container &&other) { *this = std::move(other); }

Container &Container::operator=(Container &&other) {
  if (this != &other) {
    this->shape = std::move(other.shape);
    this->data = std::move(other.data);
  }
  return *this;
}

void Container::reshape(const std::vector<int> &_shape) {
  this->shape = _shape;
  this->is_size_consistent();
}

void Container::resize(const std::vector<int> &_shape) {
  this->shape = _shape;

  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  if (size != this->data.size()) {
    this->data.resize(size);
  }
}

__global__ void container_random_fill(float *a, int size, float scale,
                               curandState *cs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    curand_init(1234, index, 0, &cs[index]);
    a[index] = (curand_uniform(&cs[index]) * 2 - 1) * scale;
  }
}

void Container::random_fill(size_t in_size, size_t out_size)
{
  float *a_ptr = RAW_PTR(this->data);
  int size = this->data.size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);

  thrust::device_vector<curandState> cs(size);
  curandState *cs_ptr = RAW_PTR(cs);
  float scale = std::sqrt((float)6) / std::sqrt((float)(in_size) + out_size);
  container_random_fill<<<grid_size, BLOCK_SIZE>>>(a_ptr, size, scale, cs_ptr);

  CUDA_POST_KERNEL_CHECK;
}

void Container::is_size_consistent() {
  int size = 1;
  for (auto it : this->shape) {
    size *= it;
  }
  CHECK_EQ(size, this->data.size(), "Container: size error");
}