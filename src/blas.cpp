#include<blas.hpp>
#include<utils.hpp>
#include<functors.hpp>

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cfloat>

void blas_add(const Container *input1, const Container *input2,
                  Container *outputs)
{
    CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
             "operator_add: size error");

    thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                      input2->get_data().begin(), outputs->get_data().begin(),
                      thrust::plus<float>());
}

void operator_add(const Container *input1, float value, Container *outputs)
{
    thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                      outputs->get_data().begin(), add_functor(value));
}

void operator_sub(const Container *input1, const Container *input2,
                  Container *outputs)
{
    CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
             "operator_sub: size error");

    thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                      input2->get_data().begin(), outputs->get_data().begin(),
                      thrust::minus<float>());
}