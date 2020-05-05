
#pragma once

#include<Layer.hpp>
#include<blas.hpp>

#ifdef DEBUG

void my_relu(const Container *input1, Container *outputs);
void my_d_relu(const Container *outputs_grad, const Container *input1,
                     Container *intputs_grad);

#endif // DEBUG

class ReLU : public Layer
{
public:
    ReLU(bool inplace) : inplace(inplace) {}

    void forward();
    void backward();

    Container *get_grad()
    {
        return this->inplace ? this->next->get_grad() : this->grad.get();
    }
    Container *get_output()
    {
        return this->inplace ? this->prev->get_output() : this->output.get();
    }

private:
    bool inplace;
};
