#pragma once

#include<blas.hpp>
#include<Optimizer.hpp>

#include <unordered_map>
#include <vector>

#ifdef DEBUG

void rmsprop_update(Container *square_grads, Container *weights,
                    const Container *grads, float learning_rate = 1e-2,
                    float l2 = 0, float beta = 0.99);

#endif // DEBUG

class RMSProp : public Optimizer
{
public:
    explicit RMSProp(float learning_rate = 0.01, float l2 = 0.001,
                     float beta = 0.99)
        : learning_rate(learning_rate), l2(l2), beta(beta)
    {
        std::cout << "learning rate: " << learning_rate << ", l2: " << l2
                  << ", beta: " << beta << std::endl;
    }

    void regist(std::vector<std::pair<Container *, Container *>> params);
    void step();

private:
    std::vector<std::unique_ptr<Container>> square_grad;

    float learning_rate;
    float l2;
    float beta;
};