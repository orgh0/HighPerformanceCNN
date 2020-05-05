#pragma once

#include <utility>
#include <vector>

#include<Container.hpp>

class Optimizer
{
public:
    virtual void step() = 0;
    virtual void regist(std::vector<std::pair<Container *, Container *>> params) = 0;

protected:
    std::vector<Container *> parameter_list;
    std::vector<Container *> grad_list;
};