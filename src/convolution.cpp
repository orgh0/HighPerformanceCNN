#include<convolution.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <memory>
#include <vector>

// C*H*W >> (C_out*k_h*k_w) * (height_col * width_col)
__global__ void my_im2col_h(const int n, const float *data_im, const int height,
                         const int width, const int kernel_h,
                         const int kernel_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int height_col, const int width_col,
                         float *data_col, int im_stride, int col_stride)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        const int batch_idx = blockIdx.y;
        data_im += batch_idx * im_stride;
        data_col += batch_idx * col_stride;

        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;

        // channel offset
        float *data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const float *data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        // copy to col
        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                        ? data_im_ptr[i * width + j]
                        : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void my_im2col(const float *data_im, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_col)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int size = channels * height_col * width_col;

    int im_stride = channels * height * width;
    int col_stride = channels * kernel_h * kernel_w * height_col * width_col;
    dim3 dim_grid(ceil((float)size / BLOCK_SIZE), batch_size);
    my_im2col_h<<<dim_grid, BLOCK_SIZE>>>(
        size, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, height_col, width_col, data_col, im_stride, col_stride);
    CUDA_POST_KERNEL_CHECK;
}

// (C_out*k_h*k_w) * (height_col * width_col) >> C*H*W
__global__ void my_col2im_h(const int n, const float *data_col, const int height,
                         const int width, const int channels,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w, const int stride_h,
                         const int stride_w, const int height_col,
                         const int width_col, float *data_im,
                         const int im_stride, const int col_stride)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        const int batch_idx = blockIdx.y;
        data_im += batch_idx * im_stride;
        data_col += batch_idx * col_stride;

        float val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);

        // compute the start and end of the col
        const int w_col_start =
            (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
        const int w_col_end = fminf(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
        const int h_col_end = fminf(h_im / stride_h + 1, height_col);

        // copy to im
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1)
        {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1)
            {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                int data_col_index =
                    (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) *
                        width_col +
                    w_col;
                val += data_col[data_col_index];
            }
        }
        data_im[index] = val;
    }
}

void my_col2im(const float *data_col, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_im)
{
    int height_col = height + 2 * pad_h - kernel_h / stride_h + 1;
    int width_col = width + 2 * pad_w - kernel_w / stride_w + 1;
    int size = channels * height * width;

    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    int im_stride = channels * height * width;
    int col_stride = channels * kernel_h * kernel_w * height_col * width_col;
    dim3 dim_grid(ceil((float)size / BLOCK_SIZE), batch_size);
    my_col2im_h<<<dim_grid, BLOCK_SIZE>>>(size, data_col, height, width, channels,
                                       kernel_h, kernel_w, pad_h, pad_w, stride_h,
                                       stride_w, height_col, width_col, data_im,
                                       im_stride, col_stride);
    CUDA_POST_KERNEL_CHECK;
}

Convolution::Convolution(int height, int width, int channel_in, int channel_out, int kernel_h,
                         int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                         bool is_bias)
    : height(height),
      width(width),
      channel_in(channel_in),
      channel_out(channel_out),
      kernel_h(kernel_h),
      kernel_w(kernel_w),
      pad_h(pad_h),
      pad_w(pad_w),
      stride_h(stride_h),
      stride_w(stride_w),
      is_bias(is_bias)
{
    int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    this->filters.reset(
        new Container({channel_out, channel_in, kernel_h, kernel_w}));
    this->filters->xavier(channel_in * height * width,
                          channel_out * height_out * width_out);
    this->filters_grad.reset(
        new Container({channel_out, channel_in, kernel_h, kernel_w}));

    if (is_bias)
    {
        this->bias.reset(new Container({1, channel_out}));
        this->bias_grad.reset(new Container({1, channel_out}));
        this->bias->xavier(channel_in * height * width,
                           channel_out * height_out * width_out);
    }
}