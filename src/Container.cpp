#include <Container.hpp>
#include <utils.hpp>

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cmath>

Container::Container(const std::vector<int> &_shape) : shape(_shape) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  this->data.resize(size);
}

Container::Container(const std::vector<int> &_shape, float value) : shape(_shape) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
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

void Container::is_size_consistent() {
  int size = 1;
  for (int i = 0; i < this->shape.size(); i++) {
    size *= this->shape[i];
  }
  CHECK_EQ(size, this->data.size(), "Container: size error");
}