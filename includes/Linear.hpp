#pragma once

#include<blas.hpp>
#include<Layer.hpp>

#include <unordered_map>

#ifdef DEBUG

void my_linear_layer(const Container *inputs, const Container *weights,
                     Container *output);

void my_d_linear_layer(
    const Container *outputs_grad, const Container *inputs, const Container *weights,
    Container *weights_grad, Container *inputs_grad,
    std::unordered_map<std::string, std::unique_ptr<Container>> &temp);

void my_linear_bias(const Container *inputs, const Container *bias,
                          Container *output);

void my_d_linear_bias(const Container *outputs_grad, Container *bias_grad);

#endif // DEBUG

class Linear : public Layer
{
public:
    explicit Linear(int in_size, int out_size, bool is_bias);

    std::vector<std::pair<Container *, Container *>> parameters();
    void forward();
    void backward();

private:
    std::unique_ptr<Container> weights;
    std::unique_ptr<Container> weights_grad;
    std::unique_ptr<Container> bias;
    std::unique_ptr<Container> bias_grad;

    std::unordered_map<std::string, std::unique_ptr<Container>> temp;

    int in_size;
    int out_size;
    bool is_bias;
};
