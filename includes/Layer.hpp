/*
Class Layer
Use: Base class for all ML layers we build.
Member Functions:
    link: Connects a layer to it's next layer
    forward: Virtual function, each layer will include it's specific Layer operation
    backward: Virtual function, each layer will include it's specific Layer operation

*/
#pragma once

#include <Container.hpp>
#include <iostream>
#include <stdexcept>

class Layer {

    public:
        Layer() {}
        Layer(const Layer &other) = delete; // Delete copy constructor
        Layer &operator=(const Layer &other) = delete;

        // link to next layer
        Layer &link(Layer &next_layer) {
            this->next = &next_layer;
            next_layer.prev = this;

            return next_layer;
        }

        virtual void forward(){
            throw std::runtime_error("Forward not implemented for this Layer");
        };

        virtual void backward(){
            throw std::runtime_error("Backward not implemented for this layer");
        };

        virtual Container *get_grad() { return this->grad.get(); }
        virtual Container *get_output() { return this->output.get(); }

    protected:
        Layer *prev;
        Layer *next;

        std::unique_ptr<Storage> grad;
        std::unique_ptr<Storage> output;
};