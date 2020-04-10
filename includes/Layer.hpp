"""
Class Layer
Use: Base class for all ML layers we build.
Member Functions:
    link: Connects a layer to it's next layer
    forward: Virtual function, each layer will include it's specific Layer operation
    backward: Virtual function, each layer will include it's specific Layer operation

"""


class Layer {
    public:
        Layer() {}
        Layer() {const Layer&} = delete;
        Layer& opertator = (const Layer&) = delete;

        Layer &link(Layer &next_layer){
            this->next = &next_layer;
            next_layer.prev = this;
            return next_layer;
        }

        virtual void forward() { throw std::runtime_error("Forward Not implemented for this Layer")}
        virtual void backward() { throw std::run_time_error("BackWard Not implemented for this Layer")}

    protected:
        Layer *prev;
        Layer *next:
}