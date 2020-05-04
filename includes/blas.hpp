#pragma once

#include<Container.hpp>
#include<utils.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void blas_add(const Container *input1, const Container *input2,
                  Container *outputs);

void blas_add(const Container *input1, float value, Container *outputs);

void blas_subtract(const Container *input1, const Container *input2,
              Container *outputs);

void blas_multiply(const Container *input1, const Container *input2,
                  Container *outputs);

void blas_multiply(const Container *input1, float value, Container *outputs);

void blas_divide(const Container *input1, const Container *input2,
                  Container *outputs);

void blas_log(const Container *input1, Container *outputs);

void blas_exp(const Container *input1, Container *outputs);

void blas_pow(const Container *input1, float e, Container *outputs);

void blas_matmul(const Container *input1, const Container *input2,
                     Container *outputs, int broadcast = 0);

void blas_transpose(const Container *input1, Container *outputs);

void blas_mean(const Container *input1, int dim, Container *outputs);

void blas_sum(const Container *input1, int dim, Container *outputs);
