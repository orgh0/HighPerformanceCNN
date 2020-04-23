#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iterator>
#include <vector>

class Container {
 public:
  explicit Container(const std::vector<int> &_shape);
  explicit Container(const std::vector<int> &_shape, float value);
  explicit Container(const std::vector<int> &_shape, const std::vector<float> &_data);

  // copy/move
  Container(const Container &other);
  Container &operator=(const Container &other);
  Container(Container &&other);
  Container &operator=(Container &&other);

  void reshape(const std::vector<int> &_shape);
  void resize(const std::vector<int> &_shape);

  // get
  std::vector<int> &get_shape() { return this->shape; };
  const std::vector<int> &get_shape() const { return this->shape; };
  thrust::device_vector<float> &get_data() { return this->data; };
  const thrust::device_vector<float> &get_data() const { return this->data; };

 private:
  void is_size_consistent();  // check data/shape size

  thrust::device_vector<float> data;
  std::vector<int> shape;
};