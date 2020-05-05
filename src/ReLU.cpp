#include<ReLU.hpp>

struct max_functor
{
    __host__ __device__ float operator()(const float &x) const
    {
        return fmaxf(0, x);
    }
};

struct relu_functor
{
    __host__ __device__ float operator()(const float &x, const float &y) const
    {
        return x > FLT_EPSILON ? y : 0;
    }
};

void my_relu(const Container *input1, Container *outputs)
{
    thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                      outputs->get_data().begin(), max_functor());
}

// Y = relu(X)
// dL/dX = relu'(X) element_mul dL/dY
void my_d_relu(const Container *outputs_grad, const Container *input1,
                     Container *intputs_grad)
{
    thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                      outputs_grad->get_data().begin(),
                      intputs_grad->get_data().begin(), relu_functor());
}

void ReLU::forward()
{
    Container *input = this->pre->get_output();

    if (this->inplace)
    {
        my_relu(input, input);
    }
    else
    {
        INIT_STORAGE(this->output, input->get_shape());
        my_relu(input, this->output.get());
    }
}

void ReLU::backward()
{
    Container *input = this->pre->get_output();
    Container *output_grad = this->next->get_grad();

    if (this->inplace)
    {
        my_d_relu(output_grad, input, output_grad);
    }
    else
    {
        INIT_STORAGE(this->grad, input->get_shape());
        my_d_relu(output_grad, input, this->grad.get());
    }
}