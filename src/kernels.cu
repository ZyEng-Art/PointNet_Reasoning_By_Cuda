#include "kernels.cuh"
__global__ void conv1d_1kernel(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias) {
    // input:: in_channels x length
    // output:: out_channels x length
    // kernel:: out_channels x in_channels
    // 如果kernel 很小，可以放入constant memory?
    // 不使用shared memory
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType sum = bias->get_element(row, 0);
    for (unsigned i = 0; i < input->height /*weight->width*/; i++) {
        sum += weight->get_element(row, i) * input->get_element(i, col);
    }
    output->set_element(row, col, sum);
}

// specially, input is n x d, weight is m x d, bias is m x 1, output is n x m
__global__ void linear(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType sum = bias->get_element(col, 0);
    for (unsigned i = 0; i < input->width /*weight->height*/; i++) {
        sum += input->get_element(row, i) * weight->get_element(col, i);
    }
    output->set_element(row, col, sum);
}

__global__ void add(Matrix *a, Matrix *b, Matrix *c) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= c->height || col >= c->width) {
        return;
    }
    c->set_element(row, col, a->get_element(row, col) + b->get_element(row, col));
}

__global__ void relu(Matrix *a) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= a->height || col >= a->width) {
        return;
    }
    a->set_element(row, col, a->get_element(row, col) > 0 ? a->get_element(row, col) : 0);
}

// TODO: 想到的优化
__global__ void batch_norm1d_3d(Matrix *input, Matrix *running_mean, Matrix *running_var, Matrix *gamma, Matrix *beta, ElementType eps) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= input->height || col >= input->width) {
        return;
    }
    ElementType mean = running_mean->get_element(row, 0);
    ElementType var = running_var->get_element(row, 0);
    ElementType x = input->get_element(row, col);
    ElementType y = (x - mean) / sqrt(var + eps);
    input->set_element(row, col, gamma->get_element(row, 0) * y + beta->get_element(row, 0));
}

__global__ void batch_norm1d_2d(Matrix *input, Matrix *running_mean, Matrix *running_var, Matrix *gamma, Matrix *beta, ElementType eps) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= input->height || col >= input->width) {
        return;
    }
    ElementType mean = running_mean->get_element(0, col);
    ElementType var = running_var->get_element(0, col);
    ElementType x = input->get_element(row, col);
    ElementType y = (x - mean) / sqrt(var + eps);
    input->set_element(row, col, gamma->get_element(0, col) * y + beta->get_element(0, col));
}
// TODO: 优化不同线程最大值
__global__ void maxpool1d_all_by_column(Matrix *input, Matrix *output) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType max = -1e9;
    for (unsigned i = 0; i < input->height; i++) {
        max = fmax(max, input->get_element(i, col));
    }
    output->set_element(row, col, max);
}

__global__ void maxpool1d_all_by_row(Matrix *input, Matrix *output) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType max = -1e9;
    for (unsigned i = 0; i < input->width; i++) {
        max = fmax(max, input->get_element(row, i));
    }
    output->set_element(row, col, max);
}

// 可能还是执行加法更快（或速度差不多)
__global__ void self_add_i(Matrix *input, unsigned I_size) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    col = col * I_size + col;
    // printf("%d %d %d %d %d\n", I_size, row, col, input->height, input->width);
    if (row >= input->height || col >= input->width) {
        return;
    }
    input->set_element(row, col, input->get_element(row, col ) + 1);
}

__global__ void transpose(Matrix *input, Matrix *output) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    output->set_element(row, col, input->get_element(col, row));
}

__global__ void multifly(Matrix *input, Matrix *output, Matrix *trans) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType sum = 0;
    for (unsigned i = 0; i < input->width /*weight->height*/; i++) {
        sum += input->get_element(row, i) * trans->get_element(i, col);
    }
    output->set_element(row, col, sum);
}

__global__ void log_softmax(Matrix *input, Matrix *output) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType sum = 0;
    for (unsigned i = 0; i < input->width; i++) {
        sum += exp(input->get_element(row, i));
    }
    output->set_element(row, col, exp(input->get_element(row, col)) / sum);
}

__global__ void arg_max(Matrix *input, Matrix *output) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType max = -1e9;
    unsigned max_index = 0;
    for (unsigned i = 0; i < input->width; i++) {
        if (input->get_element(row, i) > max) {
            max = input->get_element(row, i);
            max_index = i;
        }
    }
    output->set_element(row, col, max_index);
}