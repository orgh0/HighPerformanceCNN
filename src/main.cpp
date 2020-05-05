#include <Dataset.hpp>
#include<Model.hpp>
#include <utils.hpp>

#define BATCH_SIZE 128
#define LEARNING_RATE 0.003
#define L2 0.0001
#define EPOCHS 30
#define BETA 0.99

int main() {
  DataSet dataset("./mnist_data", true);
  dataset.forward(64, true);

  auto cudaStatus = cudaSetDevice(0);
  CHECK_EQ(cudaStatus, cudaSuccess,
           "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

  Model mnist("./mnist_data", LEARNING_RATE, L2, BETA);
  mnist.train(EPOCHS, BATCH_SIZE);
}
