#include<Loss.hpp>

#include <memory>

// L = mean(sum(-log_P element_mul Y, 1), 0)
void nll_loss(
    const Container *log_p, const Container *y, Container *output,
    std::unordered_map<std::string, std::unique_ptr<Container>> &temp)
{
    INIT_TEMP(temp, "nll_loss_batch", y->get_shape());
    blas_multiply(log_p, y, temp["nll_loss_batch"].get());

    std::vector<int> sum_shape{y->get_shape()[0], 1};
    INIT_TEMP(temp, "nll_loss_sum", sum_shape);
    blas_sum(temp["nll_loss_batch"].get(), 1, temp["nll_loss_sum"].get());

    blas_mean(temp["nll_loss_sum"].get(), 0, output);
    output->get_data()[0] *= -1;
}

// L = 1_n^T * ((-log_P element_mul Y) * 1_k) / N
// dL/d(log_P) = -Y / N
void d_nll_loss(const Container *y, Container *inputs_grad)
{
    int batch_size = *y->get_shape().begin();
    blas_multiply(y, (float)-1 / batch_size, inputs_grad);
}

void NLLLoss::forward(const Container *y)
{
    const Container *input = this->prev->get_output();
    this->y = y;

    nll_loss(input, y, this->output.get(), this->temp);
}

void NLLLoss::backward()
{
    const Container *input = this->prev->get_output();

    INIT_STORAGE(this->grad, input->get_shape());
    d_nll_loss(this->y, this->grad.get());
}
