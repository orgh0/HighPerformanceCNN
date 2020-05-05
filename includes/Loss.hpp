#pragma once

#include<blas.hpp>
#include<Layer.hpp>

#include <unordered_map>

#ifdef DEBUG

void nll_loss(
    const Container *log_p, const Container *y, Container *output,
    std::unordered_map<std::string, std::unique_ptr<Container>> &temp);

void d_nll_loss(const Container *y, Container *inputs_grad);

#endif // DEBUG

class NLLLoss : public Layer
{
public:
    NLLLoss() { this->output.reset(new Container({1, 1})); }
    void forward(const Container *y);
    void backward();

private:
    const Container *y; // backup

    std::unordered_map<std::string, std::unique_ptr<Container>> temp;
};