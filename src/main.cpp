#include <Dataset.hpp>

int main() {
  DataSet dataset("./mnist_data", true);
  dataset.forward(64, true);

  auto cudaStatus = cudaSetDevice(0);
  CHECK_EQ(cudaStatus, cudaSuccess,
           "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
}
