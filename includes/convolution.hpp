#pragma once

#include<blas.hpp>
#include<Layer.hpp>

#include <unordered_map>

class Convolution : public Layer
{
public:
    explicit Convolution(int height, int width, int channel_in, int channel_out,
                  int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                  int stride_w, bool is_bias);

    void forward();
    void backward();
    std::vector<std::pair<Storage *, Storage *>> parameters();

private:
    std::unique_ptr<Storage> filters;
    std::unique_ptr<Storage> filters_grad;
    std::unique_ptr<Storage> bias;
    std::unique_ptr<Storage> bias_grad;
    std::unique_ptr<Storage> cols;

    std::unordered_map<std::string, std::unique_ptr<Storage>> temp;

    int height;
    int width;
    int channel_in;
    int channel_out;
    int kernel_h;
    int kernel_w;
    int pad_w;
    int pad_h;
    int stride_w;
    int stride_h;
    bool is_bias;
};

#ifdef DEBUG

void my_im2col(const float *data_im, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_col);

void my_col2im(const float *data_col, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_im);

#endif