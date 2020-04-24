#include <Dataset.hpp>
#include <utils.hpp>

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>

DataSet::DataSet(std::string data_path, bool shuffle)
    : shuffle(shuffle), train_data_index(0), test_data_index(0) {
    // train data
    this->read_images(data_path + "/train-images-idx3-ubyte", this->train_data);
    this->read_labels(data_path + "/train-labels-idx1-ubyte", this->train_label);
    // test data
    this->read_images(data_path + "/t10k-images-idx3-ubyte", this->test_data);
    this->read_labels(data_path + "/t10k-labels-idx1-ubyte", this->test_label);
}

void DataSet::reset() {
    this->train_data_index = 0;
    this->test_data_index = 0;

    if (shuffle) {
    // keep random seed same
    unsigned int seed =
        std::chrono::system_clock::now().time_since_epoch().count() % 1234;

    std::shuffle(this->train_data.begin(), this->train_data.end(), std::default_random_engine(seed));
    std::shuffle(this->train_label.begin(), this->train_label.end(), std::default_random_engine(seed));
    }
}

void DataSet::forward(int batch_size, bool is_train) {
    if (is_train) {
        int start = this->train_data_index;
        int end = std::min(this->train_data_index + batch_size,
                            (int)this->train_data.size());
        this->train_data_index = end;
        int size = end - start;

        // init device memory
        std::vector<int> output_shape{size, 1, this->height, this->width};
        std::vector<int> output_label_shape{size, 10};
        INIT_STORAGE(this->output, output_shape);
        INIT_STORAGE(this->output_label, output_label_shape);
        thrust::fill(this->output_label->get_data().begin(),
                        this->output_label->get_data().end(), 0);

        // copy to device memory
        int im_stride = 1 * this->height * this->width;
        int one_hot_stride = 10;

        thrust::host_vector<
            float, thrust::system::cuda::experimental::pinned_allocator<float>>
            train_data_buffer;
        train_data_buffer.reserve(size * im_stride);

        for (int i = start; i < end; i++) {
            train_data_buffer.insert(train_data_buffer.end(),
                                    this->train_data[i].begin(),
                                    this->train_data[i].end());
            this->output_label
                ->get_data()[(i - start) * one_hot_stride + this->train_label[i]] = 1;
        }
        this->output->get_data() = train_data_buffer;

    } else {
        int start = this->test_data_index;
        int end = std::min(this->test_data_index + batch_size,
                            (int)this->test_data.size());
        this->test_data_index = end;
        int size = end - start;

        // init device memory
        std::vector<int> output_shape{size, 1, this->height, this->width};
        std::vector<int> output_label_shape{size, 10};
        INIT_STORAGE(this->output, output_shape);
        INIT_STORAGE(this->output_label, output_label_shape);
        thrust::fill(this->output_label->get_data().begin(),
                        this->output_label->get_data().end(), 0);

        // copy to device memory
        int im_stride = 1 * this->height * this->width;
        int one_hot_stride = 10;

        thrust::host_vector<
            float, thrust::system::cuda::experimental::pinned_allocator<float>>
            test_data_buffer;
        test_data_buffer.reserve(size * im_stride);

        for (int i = start; i < end; i++) {
            test_data_buffer.insert(test_data_buffer.end(),
                                    this->test_data[i].begin(),
                                    this->test_data[i].end());
            this->output_label
                ->get_data()[(i - start) * one_hot_stride + this->test_label[i]] = 1;
        }
        this->output->get_data() = test_data_buffer;
    }
}

bool DataSet::has_next(bool is_train) {
    if (is_train) {
    return this->train_data_index < this->train_data.size();
    } else {
    return this->test_data_index < this->test_data.size();
    }
}

unsigned int DataSet::reverse_int(unsigned int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255; //get last 8 bits
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) + ((unsigned int)ch3 << 8) + ch4;

}

void DataSet::read_images(std::string file_name, std::vector<std::vector<float>>& output) {
    std::ifstream file(file_name, std::ios::binary);
    if(file.is_open()) {
        unsigned int magic_number = 0;
        unsigned int num_images = 0;
        unsigned int num_rows = 0;
        unsigned int num_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_images, sizeof(num_images));
        file.read((char*)&num_rows, sizeof(num_rows));
        file.read((char*)&num_cols, sizeof(num_cols));

        magic_number = this->reverse_int(magic_number);
        num_images = this->reverse_int(num_images);
        num_rows = this->reverse_int(num_rows);
        num_cols = this->reverse_int(num_cols);

        std::cout << file_name << std::endl;
        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << num_images << std::endl;
        std::cout << "rows = " << num_rows << std::endl;
        std::cout << "cols = " << num_cols << std::endl;

        this->height = num_rows;
        this->width = num_cols;

        std::vector<unsigned char> image(num_rows * num_cols);
        std::vector<float> normalized_image(num_rows * num_cols);

        for (int i = 0; i < num_images; i++) {
            file.read((char*)&image[0], sizeof(unsigned char) * num_rows * num_cols);

            for (int i = 0; i < num_rows * num_cols; i++) {
                normalized_image[i] = (float)image[i] / 255 - 0.5;
            }
            output.push_back(normalized_image);
        }

    }
}

void DataSet::read_labels(std::string file_name, std::vector<unsigned char>& output) {
    std::ifstream file(file_name, std::ios::binary);
    if (file.is_open()) {
        unsigned int magic_number = 0;
        unsigned int num_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_images, sizeof(num_images));

        std::cout << file_name << std::endl;
        magic_number = this->reverse_int(magic_number);
        num_images = this->reverse_int(num_images);
        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << num_images << std::endl;

        for (int i = 0; i < num_images; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        output.push_back(label);
        }
    }
}
