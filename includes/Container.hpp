
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iterator>
#include <vector>

class Container {
    public:
        explicit Container(const std::vector<int> &_shape);
        explicit Container(const std::vector<int> &_shape, float value);
        explicit Container(const std::vector<int> &_shape, const std::vector<float> &_data);

        Container(const Container &other); //copy consturtor
        Container &operator=(const Container &other); //overloading = operator
        Container(Container &&other); //move constructor
        Container &operator=(Container &&other); //overloading move = operator

        std::vector<int> &get_shape() { return this-> shape; };
        thrust::device_vector<float> &get_data() { return this->data; };
        const std::vector<int> &get_shape() const { return this-> shape; };
        const thrust::device_vector<float> &get_data() const { return this->data; };

        void reshape(const std::vector<int> &_shape);
        void resize(const std::vector<int> &_shape);


    private:
        void is_size_consistent();

        thrust::device_vector<float> data; //Transfering data into the cuda container
        std::vector<int> shape; //Store shape of the data

};