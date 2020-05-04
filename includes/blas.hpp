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

void operator_mul(const Container *input1, const Container *input2,
                  Container *outputs);

void operator_mul(const Container *input1, float value, Container *outputs);


void operator_div(const Container *input1, const Container *input2,
                  Container *outputs);
