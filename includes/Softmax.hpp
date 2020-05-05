#pragma once

#include<blas.hpp>
#include<Layer.hpp>

#ifdef DEBUG

void log_softmax(const Container *input1, int dim, Container *outputs);

void d_log_softmax(const Container *output_grads, const Container *input1,
                            int dim, Container *inputs_grad);

#endif // DEBUG

class LogSoftmax : public Layer
{
public:
    explicit LogSoftmax(int dim = 1) : dim(dim) {}
    void forward();
    void backward();

private:
    int dim;
};