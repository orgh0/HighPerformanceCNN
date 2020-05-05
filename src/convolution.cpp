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

// inputs: N*C*H*W
// filters: C_out*C_in*K_h*K_w
void convolution_operator(const Container *inputs, Container *filters, Container *cols,
                   const int pad_h, const int pad_w, const int stride_h,
                   const int stride_w, Container *output) {
  CHECK_EQ(inputs->get_shape().size(), 4, "operator_conv: inputs shape error");
  CHECK_EQ(filters->get_shape().size(), 4,
           "operator_conv: filters shape error");

  int batch_size = *(inputs->get_shape().rbegin() + 3);
  int channel_in = *(inputs->get_shape().rbegin() + 2);
  int height = *(inputs->get_shape().rbegin() + 1);
  int width = *(inputs->get_shape().rbegin());

  int channel_out = *(filters->get_shape().rbegin() + 3);
  int kernel_h = *(filters->get_shape().rbegin() + 1);
  int kernel_w = *(filters->get_shape().rbegin());

  CHECK_EQ(*(filters->get_shape().rbegin() + 2), channel_in,
           "operator_conv: channel size error");

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  // im2col
  // [batch_size*(C_in*k_h*k_w)*(height_col * width_col)]
  const float *inputs_ptr = RAW_PTR(inputs->get_data());
  const float *filters_ptr = RAW_PTR(filters->get_data());
  float *cols_ptr = RAW_PTR(cols->get_data());
  my_im2col(inputs_ptr, batch_size, channel_in, height, width, kernel_h, kernel_w,
         pad_h, pad_w, stride_h, stride_w, cols_ptr);

  // Y = F * col
  // [C_out*(C_in*k_h*k_w)] * [batch_size *
  // (C_in*k_h*k_w)*(height_col*width_col)] = [batch_size * channel_out *
  // (height_col * width_col)]
  filters->reshape({channel_out, channel_in * kernel_h * kernel_w});
  blas_matmul(filters, cols, output, 1);  // broadcast param 1

  // recover shapre
  filters->reshape({channel_out, channel_in, kernel_h, kernel_w});
}

// Y = F * col
// dL/dF = dL/dY * col^T
// dL/d_col = F^T * dL/dY
// dL/d_im = col2im(dL/d_col)
void d_convolution_operator(
    Container *outputs_grad, const Container *inputs, const Container *cols,
    Container *filters, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Container *filters_grad, Container *inputs_grad,
    std::unordered_map<std::string, std::unique_ptr<Container>> &temp) {
  CHECK_EQ(outputs_grad->get_shape().size(), 4,
           "operator_d_conv: outputs_grad shape error");
  CHECK_EQ(inputs->get_shape().size(), 4,
           "operator_d_conv: inputs shape error");
  CHECK_EQ(cols->get_shape().size(), 3, "operator_d_conv: cols shape error");
  CHECK_EQ(filters->get_shape().size(), 4,
           "operator_d_conv: filters shape error");

  int batch_size = *(inputs->get_shape().rbegin() + 3);
  int channel_in = *(inputs->get_shape().rbegin() + 2);
  int height = *(inputs->get_shape().rbegin() + 1);
  int width = *(inputs->get_shape().rbegin());

  int channel_out = *(filters->get_shape().rbegin() + 3);
  int kernel_h = *(filters->get_shape().rbegin() + 1);
  int kernel_w = *(filters->get_shape().rbegin());

  CHECK_EQ(*(filters->get_shape().rbegin() + 2), channel_in,
           "operator_d_conv: channel size error");

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  // dL/dY reshape
  outputs_grad->reshape({batch_size, channel_out, height_col * width_col});

  // col^T
  std::vector<int> cols_t_shape(
      {batch_size, height_col * width_col, channel_in * kernel_h * kernel_w});
  INIT_TEMP(temp, "cols_t", cols_t_shape);
  blas_transpose(cols, temp["cols_t"].get());  // last two dims transpose

  // dL/dF = dL/dY * col^T
  std::vector<int> dl_df_shape{batch_size, channel_out,
                               channel_in * kernel_h * kernel_w};
  INIT_TEMP(temp, "dl_df", dl_df_shape);
  blas_matmul(outputs_grad, temp["cols_t"].get(),
                  temp["dl_df"].get());                // last two dims matmul
  blas_sum(temp["dl_df"].get(), 0, filters_grad);  // sum along batch

  // F^T
  std::vector<int> filters_t_shape{channel_in * kernel_h * kernel_w,
                                   channel_out};
  INIT_TEMP(temp, "filters_t", filters_t_shape);
  filters->reshape(
      {channel_out, channel_in * kernel_h * kernel_w});  // filters reshape
  blas_transpose(filters, temp["filters_t"].get());
  filters->reshape(
      {channel_out, channel_in, kernel_h, kernel_w});  // filters recover

  // dL/d_col = F^T * dL/dY
  std::vector<int> dl_dcol_shape{batch_size, channel_in * kernel_h * kernel_w,
                                 height_col * width_col};
  INIT_TEMP(temp, "dl_dcol", dl_dcol_shape);
  blas_matmul(temp["filters_t"].get(), outputs_grad, temp["dl_dcol"].get(),
                  1);  // broadcast param 1

  // dL/dY recover
  outputs_grad->reshape({batch_size, channel_out, height_col, width_col});

  // dL/d_im = col2im(dL/d_col)
  float *dl_dcol_ptr = RAW_PTR(temp["dl_dcol"].get()->get_data());
  float *inputs_grad_ptr = RAW_PTR(inputs_grad->get_data());
  my_col2im(dl_dcol_ptr, batch_size, channel_in, height, width, kernel_h, kernel_w,
         pad_h, pad_w, stride_h, stride_w, inputs_grad_ptr);
}

__global__ void convolution_bias_operator_h(const float *inputs, const float *bias,
                                     float *output, int channel_size,
                                     int channel_stride, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int col = (index / channel_stride) % channel_size;
    output[index] = inputs[index] + bias[col];
  }
}

void convolution_bias_operator(const Container *inputs, const Container *bias,
                        Container *output) {
  CHECK_EQ(bias->get_data().size(), *(inputs->get_shape().begin() + 1),
           "operator_conv_bias: size error");

  const float *inputs_ptr = RAW_PTR(inputs->get_data());
  const float *bias_ptr = RAW_PTR(bias->get_data());
  float *output_ptr = RAW_PTR(output->get_data());

  int channel_in = *(inputs->get_shape().rbegin() + 2);
  int height = *(inputs->get_shape().rbegin() + 1);
  int width = *(inputs->get_shape().rbegin());

  int size = inputs->get_data().size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  convolution_bias_operator_h<<<grid_size, BLOCK_SIZE>>>(
      inputs_ptr, bias_ptr, output_ptr, channel_in, height * width, size);

  CUDA_POST_KERNEL_CHECK;
}

void d_convolution_bias_operator(
    const Container *outputs_grad, Container *bias_grad,
    std::unordered_map<std::string, std::unique_ptr<Container>> &temp) {
  // N*C*H*W ==> 1*C
  int batch_size = outputs_grad->get_shape()[0];
  int channels = outputs_grad->get_shape()[1];
  int height = outputs_grad->get_shape()[2];

  // reduce W
  std::vector<int> sum3_shape{batch_size, channels, height};
  INIT_TEMP(temp, "sum3", sum3_shape);
  blas_sum(outputs_grad, 3, temp["sum3"].get());

  // reduce H
  std::vector<int> sum2_shape({batch_size, channels});
  INIT_TEMP(temp, "sum2", sum2_shape);
  blas_sum(temp["sum3"].get(), 2, temp["sum2"].get());

  // reduce N
  blas_sum(temp["sum2"].get(), 0, bias_grad);
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

std::vector<std::pair<Container *, Container *>> Convolution::parameters()
{
    if (this->is_bias)
    {
        return {{this->filters.get(), this->filters_grad.get()},
                {this->bias.get(), this->bias_grad.get()}};
    }
    else
    {
        return {{this->filters.get(), this->filters_grad.get()}};
    }
}

void Convolution::forward()
{
    const Container *input = this->prev->get_output();
    int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    std::vector<int> output_shape{input->get_shape()[0], this->channel_out,
                                  height_out, width_out};
    std::vector<int> cols_shape{input->get_shape()[0],
                                channel_in * kernel_h * kernel_w,
                                height_out * width_out};

    INIT_STORAGE(this->output, output_shape);
    INIT_STORAGE(this->cols, cols_shape);

    convolution_operator(input, this->filters.get(), this->cols.get(), pad_h, pad_w,
                  stride_h, stride_w, this->output.get());

    if (this->bias)
    {
        convolution_bias_operator(this->output.get(), this->bias.get(),
                           this->output.get());
    }
}

void Convolution::backward()
{
    const Container *input = this->prev->get_output();
    Container *output_grad = this->next->get_grad();

    int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    INIT_STORAGE(this->grad, input->get_shape());

    if (this->bias)
    {
        d_convolution_bias_operator(output_grad, this->bias_grad.get(), this->temp);
    }

    d_convolution_operator(output_grad, input, this->cols.get(), this->filters.get(),
                    pad_h, pad_w, stride_h, stride_w, this->filters_grad.get(),
                    this->grad.get(), this->temp);
}
