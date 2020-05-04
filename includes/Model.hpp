#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include<Dataset.hpp>
#include<blas.hpp>

class Model {
public:
    explicit Model(std::string dataset_path, float learning_rate, float l2,
                    float beta); // constructor for Model class
    
    void train(int epochs, int batch_size); // neural network train
    void test(int batch_size); // neural network forward

private:
    void forward(int batch_size, bool is_train); // neural network forward
    void backward();                             // neural network backward

    std::unique_ptr<DataSet> dataset;
}