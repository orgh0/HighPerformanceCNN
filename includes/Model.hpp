#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include<Dataset.hpp>
#include<blas.hpp>
#include<RMSProp.hpp>
#include<convolution.hpp>
#include<ReLU.hpp>
#include<Linear.hpp>
#include<Maxpool.hpp>
#include<Flatten.hpp>
#include<Softmax.hpp>
#include<Loss.hpp>

class Model {
public:
    explicit Model(std::string dataset_path, float learning_rate, float l2,
                    float beta); // constructor for Model class
    
    void train(int epochs, int batch_size); // neural network train
    void test(int batch_size); // neural network forward

private:
    void forward(int batch_size, bool is_train); // neural network forward
    void backward();                             // neural network backward

    std::pair<int, int> accuracy(
        const thrust::host_vector<
            float, thrust::system::cuda::experimental::pinned_allocator<float>> &
            probs,
        int cls_size,
        const thrust::host_vector<
            float, thrust::system::cuda::experimental::pinned_allocator<float>> &
            labels); // top1_accuracy

    std::unique_ptr<DataSet> dataset;
    std::unique_ptr<RMSProp> rmsprop;

    std::unique_ptr<Convolution> conv1;
    std::unique_ptr<ReLU> conv1_relu;
    std::unique_ptr<MaxPool> max_pool1;

    std::unique_ptr<Convolution> conv2;
    std::unique_ptr<ReLU> conv2_relu;
    std::unique_ptr<MaxPool> max_pool2;

    std::unique_ptr<Convolution> conv3;
    std::unique_ptr<ReLU> conv3_relu;
    std::unique_ptr<Flatten> flatten;

    std::unique_ptr<Linear> fc1;
    std::unique_ptr<ReLU> fc1_relu;

    std::unique_ptr<Linear> fc2;
    std::unique_ptr<ReLU> fc2_relu;
    std::unique_ptr<LogSoftmax> log_softmax;
    std::unique_ptr<NLLLoss> nll_loss;
};