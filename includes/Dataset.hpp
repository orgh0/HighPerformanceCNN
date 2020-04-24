#pragma once

#include <Layer.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class DataSet : public Layer {
 public:
  explicit DataSet(std::string data_path, bool shuffle = false); //constructor
  void reset(); //setting train indexes to zero

  void forward(int batch_size, bool is_train); //get new batch and push it to model
  bool has_next(bool is_train); //check is there are more batches left

  int get_height() { return this->height; }
  int get_width() { return this->width; }
  Container* get_label() { return this->output_label.get(); }

 private:
  unsigned int reverse_int(unsigned int i);  // big endian
  void read_images(std::string file_name, std::vector<std::vector<float>>& output); //read dataset file and store images in output
  void read_labels(std::string file_name, std::vector<unsigned char>& output); //read dataset file and store labels in output

  std::vector<std::vector<float>> train_data;
  std::vector<unsigned char> train_label;
  int train_data_index;

  std::vector<std::vector<float>> test_data;
  std::vector<unsigned char> test_label;
  int test_data_index;

  int height;
  int width;
  bool shuffle;
  std::unique_ptr<Container> output_label;
};
