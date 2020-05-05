#pragma once

#include<Layer.hpp>
#include<Container.hpp>

class Flatten : public Layer
{
public:
    Flatten(bool inplace) : inplace(inplace) {}

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
    std::vector<int> in_shape;
    bool inplace;
};