#include "kernels.cuh"
__global__ void conv1d_1kernel(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias) {
    // input:: in_channels x length
    // output:: out_channels x length
    // kernel:: out_channels x in_channels
    // 如果kernel 很小，可以放入constant memory?
    // 不使用shared memory
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    ElementType sum = bias->get_element(row, 0);
    for (unsigned i = 0; i < input->height /*weight->width*/; i++) {
        // printf("%d %d %d  %f %f\n", row, col, i, weight->get_element(row, i), input->get_element(i, col));
        sum += weight->get_element(row, i) * input->get_element(i, col);
    }
    // printf("%d %d %f\n", row, col, sum);
    output->set_element(row, col, sum);
}

__global__ void conv1d_1kernel_shared_mem(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias) {
    // input:: in_channels x length
    // output:: out_channels x length
    // kernel:: out_channels x in_channels
    // 使用shared memory
    __shared__ ElementType shared_input[32][32];
    __shared__ ElementType shared_weight[32][32];
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    // matrix 都是按32对齐了，不用进行判断
    ElementType sum = bias->get_element(row, 0);
    for (unsigned i = 0; i < input->height /*weight->width*/; i += blockDim.x) {
        shared_input[threadIdx.y][threadIdx.x] = 0;
        shared_weight[threadIdx.y][threadIdx.x] = 0;
        if (i + threadIdx.y < input->height && col < input->width)
            shared_input[threadIdx.y][threadIdx.x] = input->get_element(i + threadIdx.y, col);
        if (row < weight->height && i + threadIdx.x < weight->width)
            shared_weight[threadIdx.y][threadIdx.x] = weight->get_element(row, i + threadIdx.x);
        __syncthreads();
        for (unsigned j = 0; j < blockDim.x; j++) {
            sum += shared_weight[threadIdx.y][j] * shared_input[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row >= output->height || col >= output->width) {
        return;
    }
    output->set_element(row, col, sum);
}

// specially, input is n x d, weight is m x d, bias is m x 1, output is n x m
__global__ void linear(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
       if(row >= output->height || col >= output->width)
        return;
    ElementType sum = bias->get_element(col, 0);
    for (unsigned i = 0; i < input->width /*weight->height*/; i++) {
        sum += input->get_element(row, i) * weight->get_element(col, i);
    }
    output->set_element(row, col, sum);
}

__global__ void linear_shared_mem(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias) {
    __shared__ ElementType shared_input[32][32];
    __shared__ ElementType shared_weight[32][32];
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    ElementType sum = bias->get_element(col, 0);
    for (unsigned i = 0; i < input->width /*weight->height*/; i+=blockDim.x) {
        shared_input[threadIdx.y][threadIdx.x] = 0;
        shared_weight[threadIdx.y][threadIdx.x] = 0;
        // printf("%d %d %f %f\n", threadIdx.y, threadIdx.x, input->get_element(row, i + threadIdx.x), weight->get_element(col, i + threadIdx.x));
        if (i + threadIdx.x < input->width){
            if (row < input->height)
                shared_input[threadIdx.y][threadIdx.x] = input->get_element(row, i + threadIdx.x);
            if (col < weight->height)
                shared_weight[threadIdx.y][threadIdx.x] = weight->get_element(col, i + threadIdx.y);
        }
        __syncthreads();
        for (unsigned j = 0; j < blockDim.x; j++) {
            // printf("%d %d %d %f %f\n", threadIdx.x, threadIdx.y, j, shared_weight[threadIdx.x][j], shared_input[threadIdx.y][j]);
            sum += shared_weight[j][threadIdx.x] * shared_input[threadIdx.y][j];
        }
        __syncthreads();
    }
    if(row >= output->height || col >= output->width)
        return;
    output->set_element(row, col, sum);
}

__global__ void add(Matrix *a, Matrix *b, Matrix *c) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    c->set_element(row, col, a->get_element(row, col) + b->get_element(row, col));
}

__global__ void relu(Matrix *a) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= a->height || col >= a->width)
        return;
    a->set_element(row, col, a->get_element(row, col) > 0 ? a->get_element(row, col) : 0);
}

// TODO: 想到的优化
__global__ void batch_norm1d_3d(Matrix *input, Matrix *running_mean, Matrix *running_var, Matrix *gamma, Matrix *beta, ElementType eps) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
       if(row >= input->height || col >= input->width)
        return;
    ElementType mean = running_mean->get_element(row, 0);
    ElementType var = running_var->get_element(row, 0);
    ElementType x = input->get_element(row, col);
    ElementType y = (x - mean) / sqrt(var + eps);
    input->set_element(row, col, gamma->get_element(row, 0) * y + beta->get_element(row, 0));
}

__global__ void batch_norm1d_2d(Matrix *input, Matrix *running_mean, Matrix *running_var, Matrix *gamma, Matrix *beta, ElementType eps) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
       if(row >= input->height || col >= input->width)
        return;
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
    ElementType max = -1e9;
    for (unsigned i = 0; i < input->height; i++) {
        max = fmax(max, input->get_element(i, col));
    }
    output->set_element(row, col, max);
}

__global__ void maxpool1d_all_by_row(Matrix *input, Matrix *output) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= output->height || col >= output->width)
        return;
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
    if(row >= input->height || col >= input->width)
        return;
    // printf("%d %d %d %d %d\n", I_size, row, col, input->height, input->width);
    input->set_element(row, col, input->get_element(row, col) + 1);
}

__global__ void transpose(Matrix *input, Matrix *output) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= output->height || col >= output->width)
        return;
    output->set_element(row, col, input->get_element(col, row));
}

__global__ void multifly(Matrix *input, Matrix *output, Matrix *trans) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= output->height || col >= output->width)
        return;
    ElementType sum = 0;
    for (unsigned i = 0; i < input->width /*weight->height*/; i++) {
        sum += input->get_element(row, i) * trans->get_element(i, col);
    }
    output->set_element(row, col, sum);
}

// input d x n, trans d x m, output m x n (input^T * trans)^T=trans^T * input
__global__ void multifly_with_t_shared_mem(Matrix *input, Matrix *output, Matrix *trans) {
   __shared__ ElementType shared_input[32][32];
    __shared__ ElementType shared_trans[32][32];
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    // matrix 都是按32对齐了，不用进行判断
    ElementType sum = 0;
    for (unsigned i = 0; i < input->height /*weight->width*/; i += blockDim.x) {
        shared_input[threadIdx.y][threadIdx.x] = 0;
        shared_trans[threadIdx.y][threadIdx.x] = 0;
        if (i + threadIdx.y < input->height && col < input->width)
            shared_input[threadIdx.y][threadIdx.x] = input->get_element(i + threadIdx.y, col);
        if (row < trans->width && i + threadIdx.x < trans->height)
            shared_trans[threadIdx.y][threadIdx.x] = trans->get_element(i + threadIdx.x, row);
        __syncthreads();
        for (unsigned j = 0; j < blockDim.x; j++) {
            sum += shared_trans[threadIdx.y][j] * shared_input[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row >= output->height || col >= output->width) {
        return;
    }
    output->set_element(row, col, sum);
}

// input d x n, trans d x m, output m x n (input^T * trans)^T=trans^T * input
__global__ void multifly_with_t(Matrix *input, Matrix *output, Matrix *trans) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= output->height || col >= output->width)
        return;
    ElementType sum = 0;
    for (unsigned i = 0; i < input->height; i++) {
        sum += input->get_element(i, col) * trans->get_element(i, row);
    }
    output->set_element(row, col, sum);
}

// nonuse
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