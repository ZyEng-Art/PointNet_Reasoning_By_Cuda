// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include <cassert>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <hdf5/serial/H5Cpp.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <unistd.h>
typedef float ElementType;
class Matrix {
private:
public:
    ElementType *data;
    unsigned dim;
    unsigned batch;
    unsigned height;
    unsigned width;

    __device__ ElementType get_element(unsigned row, unsigned col) {
        return this->data[row * this->width + col];
    }
    __device__ ElementType get_element(unsigned batch, unsigned row, unsigned col) {
        return this->data[batch * this->height * this->width + row * this->width + col];
    }
    __device__ void set_element(unsigned row, unsigned col, ElementType element) {
        this->data[row * this->width + col] = element;
    }
    __device__ void set_element(unsigned batch, unsigned row, unsigned col, ElementType element) {
        this->data[batch * this->height * this->width + row * this->width + col] = element;
    }
    unsigned size() {
        if (dim == 2)
            return height * width;
        return height * width * batch;
    }
    Matrix(ElementType *data, unsigned height, unsigned width) {
        this->data = data;
        this->height = height;
        this->width = width;
    }
    void reshape(unsigned height, unsigned width) {
        assert(this->size() == height * width);
        this->height = height;
        this->width = width;
        this->dim = 2;
    }
    void reshape(unsigned batch, unsigned height, unsigned width) {
        assert(this->size() == height * width * batch);
        this->height = height;
        this->width = width;
        this->batch = batch;
        this->dim = 3;
    }
    void dump() {
        if (this->dim == 2) {
            std::cout << "size:(" << height << "," << width << ")" << std::endl;
            float *output = new float[true_height * true_width];
            cudaMemcpy(output, data, true_height * true_width * sizeof(ElementType), cudaMemcpyDeviceToHost);
            for (int i = 0; i < height && i < 5; i++) {
                std::cout << "row " << i << ": ";
                for (int j = 0; j < width && j < 5; j++) {
                    std::cout << output[i * true_width + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "finish" << std::endl;
            delete output;
            return;
        }
        std::cout << "size:(" << batch << "," << height << "," << width << ")" << std::endl;
        float *output = new float[batch * true_height * true_width];
        cudaMemcpy(output, data, batch * true_height * true_width * sizeof(ElementType), cudaMemcpyDeviceToHost);
        for (int k = 0; k < batch && k < 5; k++) {
            std::cout << "batch " << k << std::endl;
            for (int i = 0; i < height && i < 5; i++) {
                std::cout << "     row " << i << ": ";
                for (int j = 0; j < width && j < 5; j++) {
                    std::cout << output[i * true_width + j] << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "finish" << std::endl;
        delete output;
    }
    ~Matrix() {
        if (data)
            cudaFree(data);
    }
};

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
    input->set_element(row, col, input->get_element(row, col) + 1);
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

Matrix *new_unified_matrix(unsigned height, unsigned width) {
    ElementType *device_array;
    cudaMalloc(&device_array, height * width * sizeof(ElementType));
    Matrix *unified_matrix;
    cudaMallocManaged(&unified_matrix, sizeof(Matrix));
    unified_matrix->dim = 2;
    unified_matrix->height = height;
    unified_matrix->width = width;
    unified_matrix->data = device_array;
    return unified_matrix;
}

Matrix *new_unified_matrix(unsigned batch, unsigned height, unsigned width) {
    ElementType *device_array;
    cudaError_t err = cudaMalloc(&device_array, batch * height * width * sizeof(ElementType));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        exit(-1);
    }
    Matrix *unified_matrix;
    cudaMallocManaged(&unified_matrix, sizeof(Matrix));
    unified_matrix->dim = 3;
    unified_matrix->batch = batch;
    unified_matrix->height = height;
    unified_matrix->width = width;
    unified_matrix->data = device_array;
    return unified_matrix;
}

Matrix *host_device_by_matrix(std::vector<ElementType> &host, unsigned height, unsigned width) {
    assert(host.size() == height * width);
    Matrix *device_matrix = new_unified_matrix(height, width);
    cudaMemcpy(device_matrix->data, host.data(), host.size() * sizeof(ElementType), cudaMemcpyHostToDevice);
    return device_matrix;
}

Matrix *host_device_by_matrix(std::vector<ElementType> &host, unsigned batch, unsigned height, unsigned width) {
    assert(host.size() == height * width * batch);
    Matrix *device_matrix = new_unified_matrix(batch, height, width);
    cudaMemcpy(device_matrix->data, host.data(), host.size() * sizeof(ElementType), cudaMemcpyHostToDevice);
    return device_matrix;
}

void free_matrix(Matrix *matrix) {
    matrix->~Matrix();
    cudaFree(matrix);
}

// input 在设备内存, (B, in_channels, N)
// 输出(B, out_channels, N)
Matrix *Conv1d(Matrix *input, std::vector<ElementType> &weight, std::vector<ElementType> &bias, unsigned in_channels, unsigned out_channels) {
    assert(input->dim == 3);
    assert(weight.size() == in_channels * out_channels);
    assert(bias.size() == out_channels);
    assert(input->height == in_channels);
    Matrix *weights = host_device_by_matrix(weight, out_channels, in_channels);
    Matrix *biases = host_device_by_matrix(bias, out_channels, 1);
    Matrix *output = new_unified_matrix(input->batch, out_channels, input->width);

    dim3 block_size(32, 32);
    dim3 grid_size((output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y);
    conv1d_1kernel<<<grid_size, block_size>>>(input, output, weights, biases);
    free_matrix(weights);
    free_matrix(biases);
    return output;
}

Matrix *Linear(Matrix *input, std::vector<ElementType> &weight, std::vector<ElementType> &bias, unsigned in_features, unsigned out_features) {
    assert(input->dim == 2);
    assert(weight.size() == in_features * out_features);
    assert(bias.size() == out_features);
    assert(input->width == in_features);
    Matrix *weights = host_device_by_matrix(weight, out_features, in_features);
    Matrix *biases = host_device_by_matrix(bias, out_features, 1);
    Matrix *output = new_unified_matrix(input->height, out_features);

    dim3 block_size(32, 32);
    dim3 grid_size((output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y);
    linear<<<grid_size, block_size>>>(input, output, weights, biases);
    free_matrix(weights);
    free_matrix(biases);
    return output;
}

void Relu(Matrix *input) {
    dim3 block_size(32, 32);
    dim3 grid_size((input->width + block_size.x - 1) / block_size.x,
        (input->height + block_size.y - 1) / block_size.y);
    relu<<<grid_size, block_size>>>(input);
}

Matrix *Transpose(Matrix *input) {
    assert(input->dim == 3);
    Matrix *output = new_unified_matrix(input->batch, input->width, input->height);
    dim3 block_size(32, 32);
    dim3 grid_size((output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y);
    transpose<<<grid_size, block_size>>>(input, output);
    return output;
}

Matrix *Max(Matrix *input) {
    assert(input->dim == 3);
    Matrix *output = new_unified_matrix(input->height, 1);
    dim3 block_size(1024, 1);
    dim3 grid_size((output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y);
    maxpool1d_all_by_row<<<grid_size, block_size>>>(input, output);
    return output;
}

void BatchNorm1d_2d(Matrix *input, std::vector<ElementType> &running_mean, std::vector<ElementType> &running_var, std::vector<ElementType> &gamma, std::vector<ElementType> &beta, ElementType eps) {
    assert(input->dim == 2);
    assert(running_mean.size() == input->width);
    assert(running_var.size() == input->width);
    assert(gamma.size() == input->width);
    assert(beta.size() == input->width);
    Matrix *running_means = host_device_by_matrix(running_mean, 1, input->width);
    Matrix *running_vars = host_device_by_matrix(running_var, 1, input->width);
    Matrix *gammas = host_device_by_matrix(gamma, 1, input->width);
    Matrix *betas = host_device_by_matrix(beta, 1, input->width);

    dim3 block_size(32, 32);
    dim3 grid_size((input->width + block_size.x - 1) / block_size.x,
        (input->height + block_size.y - 1) / block_size.y);

    batch_norm1d_2d<<<grid_size, block_size>>>(input, running_means, running_vars, gammas, betas, eps);
    free_matrix(running_means);
    free_matrix(running_vars);
    free_matrix(gammas);
    free_matrix(betas);
}
void BatchNorm1d_3d(Matrix *input, std::vector<ElementType> &running_mean, std::vector<ElementType> &running_var, std::vector<ElementType> &gamma, std::vector<ElementType> &beta, ElementType eps) {
    assert(input->dim == 3);
    assert(running_mean.size() == input->height);
    assert(running_var.size() == input->height);
    assert(gamma.size() == input->height);
    assert(beta.size() == input->height);
    Matrix *running_means = host_device_by_matrix(running_mean, input->height, 1);
    Matrix *running_vars = host_device_by_matrix(running_var, input->height, 1);
    Matrix *gammas = host_device_by_matrix(gamma, input->height, 1);
    Matrix *betas = host_device_by_matrix(beta, input->height, 1);

    dim3 block_size(32, 32);
    dim3 grid_size((input->width + block_size.x - 1) / block_size.x,
        (input->height + block_size.y - 1) / block_size.y);
    batch_norm1d_3d<<<grid_size, block_size>>>(input, running_means, running_vars, gammas, betas, eps);
    free_matrix(running_means);
    free_matrix(running_vars);
    free_matrix(gammas);
    free_matrix(betas);
}

void Self_Add_I(Matrix *input) {
    assert(input->dim == 2);
    unsigned I_size = sqrt(input->width);
    assert(input->width == I_size * I_size);
    dim3 block_size(32, 32); // 加入 batch后可能有所不同
    dim3 grid_size((I_size + block_size.x - 1) / block_size.x,
        (input->height + block_size.y - 1) / block_size.y);
    self_add_i<<<grid_size, block_size>>>(input, I_size);
}

Matrix *STN3d(Matrix *input, std::map<std::string, std::vector<ElementType>> &params, ElementType eps = 1e-5) {
    // input->dump();
    Matrix *conv1 = Conv1d(input, params["feat.stn.conv1.weight"], params["feat.stn.conv1.bias"], 3, 64);
    // conv1->dump();
    BatchNorm1d_3d(conv1, params["feat.stn.bn1.running_mean"], params["feat.stn.bn1.running_var"], params["feat.stn.bn1.weight"], params["feat.stn.bn1.bias"], eps);
    // conv1->dump();
    Relu(conv1);
    // conv1->dump();
    Matrix *conv2 = Conv1d(conv1, params["feat.stn.conv2.weight"], params["feat.stn.conv2.bias"], 64, 128);
    BatchNorm1d_3d(conv2, params["feat.stn.bn2.running_mean"], params["feat.stn.bn2.running_var"], params["feat.stn.bn2.weight"], params["feat.stn.bn2.bias"], eps);
    Relu(conv2);
    free_matrix(conv1);
    Matrix *conv3 = Conv1d(conv2, params["feat.stn.conv3.weight"], params["feat.stn.conv3.bias"], 128, 1024);
    BatchNorm1d_3d(conv3, params["feat.stn.bn3.running_mean"], params["feat.stn.bn3.running_var"], params["feat.stn.bn3.weight"], params["feat.stn.bn3.bias"], eps);
    Relu(conv3);
    free_matrix(conv2);
    // conv3->dump();
    Matrix *max_pool = Max(conv3); //(B, 1024, 1)
    free_matrix(conv3);

    // max_pool->dump();
    max_pool->reshape(1, 1024); // (B, 1024)
    Matrix *fc1 = Linear(max_pool, params["feat.stn.fc1.weight"], params["feat.stn.fc1.bias"], 1024, 512);
    // fc1->dump();
    BatchNorm1d_2d(fc1, params["feat.stn.bn4.running_mean"], params["feat.stn.bn4.running_var"], params["feat.stn.bn4.weight"], params["feat.stn.bn4.bias"], eps);
    Relu(fc1);
    // fc1->dump();
    free_matrix(max_pool);
    Matrix *fc2 = Linear(fc1, params["feat.stn.fc2.weight"], params["feat.stn.fc2.bias"], 512, 256);
    BatchNorm1d_2d(fc2, params["feat.stn.bn5.running_mean"], params["feat.stn.bn5.running_var"], params["feat.stn.bn5.weight"], params["feat.stn.bn5.bias"], eps);
    Relu(fc2);
    // fc2->dump();
    free_matrix(fc1);
    Matrix *fc3 = Linear(fc2, params["feat.stn.fc3.weight"], params["feat.stn.fc3.bias"], 256, 9);
    free_matrix(fc2);

    // fc3->dump();
    Self_Add_I(fc3);
    // fc3->dump();
    cudaDeviceSynchronize();
    fc3->reshape(input->batch, 3, 3);
    // fc3->dump();
    return fc3;
}

// input: (B, k, N)
// output: (B, k, k)
Matrix *STNkd(Matrix *input, std::map<std::string, std::vector<ElementType>> &params, unsigned k) {
    // input->dump();
    Matrix *conv1 = Conv1d(input, params["feat.fstn.conv1.weight"], params["feat.fstn.conv1.bias"], k, 64);
    BatchNorm1d_3d(conv1, params["feat.fstn.bn1.running_mean"], params["feat.fstn.bn1.running_var"], params["feat.fstn.bn1.weight"], params["feat.fstn.bn1.bias"], 1e-5);
    Relu(conv1);
    // conv1->dump();
    Matrix *conv2 = Conv1d(conv1, params["feat.fstn.conv2.weight"], params["feat.fstn.conv2.bias"], 64, 128);
    BatchNorm1d_3d(conv2, params["feat.fstn.bn2.running_mean"], params["feat.fstn.bn2.running_var"], params["feat.fstn.bn2.weight"], params["feat.fstn.bn2.bias"], 1e-5);
    Relu(conv2);
    free_matrix(conv1);
    Matrix *conv3 = Conv1d(conv2, params["feat.fstn.conv3.weight"], params["feat.fstn.conv3.bias"], 128, 1024);
    BatchNorm1d_3d(conv3, params["feat.fstn.bn3.running_mean"], params["feat.fstn.bn3.running_var"], params["feat.fstn.bn3.weight"], params["feat.fstn.bn3.bias"], 1e-5);
    Relu(conv3);
    free_matrix(conv2);

    Matrix *max_pool = Max(conv3);
    free_matrix(conv3);

    max_pool->reshape(1, 1024);
    Matrix *fc1 = Linear(max_pool, params["feat.fstn.fc1.weight"], params["feat.fstn.fc1.bias"], 1024, 512);
    BatchNorm1d_2d(fc1, params["feat.fstn.bn4.running_mean"], params["feat.fstn.bn4.running_var"], params["feat.fstn.bn4.weight"], params["feat.fstn.bn4.bias"], 1e-5);
    Relu(fc1);
    free_matrix(max_pool);
    Matrix *fc2 = Linear(fc1, params["feat.fstn.fc2.weight"], params["feat.fstn.fc2.bias"], 512, 256);
    BatchNorm1d_2d(fc2, params["feat.fstn.bn5.running_mean"], params["feat.fstn.bn5.running_var"], params["feat.fstn.bn5.weight"], params["feat.fstn.bn5.bias"], 1e-5);
    Relu(fc2);
    free_matrix(fc1);
    Matrix *fc3 = Linear(fc2, params["feat.fstn.fc3.weight"], params["feat.fstn.fc3.bias"], 256, k * k);
    free_matrix(fc2);

    Self_Add_I(fc3);
    cudaDeviceSynchronize(); // 不加这句话后面这个会先运行，影响执行
    fc3->reshape(input->batch, k, k);
    return fc3;
}

Matrix *Multifly(Matrix *input, Matrix *weight) {
    assert(input->dim == 3);
    assert(input->width == weight->height);
    Matrix *output = new_unified_matrix(input->batch, input->height, weight->width);
    dim3 block_size(32, 32);
    dim3 grid_size((output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y);
    multifly<<<grid_size, block_size>>>(input, output, weight);
    return output;
}

// input: (B, 3, N)
Matrix *PointNetEncoder(Matrix *input, std::map<std::string, std::vector<ElementType>> &params) {
    Matrix *trans = STN3d(input, params); // (B, 3, 3)
    // trans->dump();
    Matrix *trans_t = Transpose(trans);
    Matrix *after_trans = Multifly(trans_t, input); // (B, 3, N)
    // after_trans->dump();
    free_matrix(trans);
    free_matrix(trans_t);

    Matrix *conv1 = Conv1d(after_trans, params["feat.conv1.weight"], params["feat.conv1.bias"], 3, 64); //(B, 64, N)
    BatchNorm1d_3d(conv1, params["feat.bn1.running_mean"], params["feat.bn1.running_var"], params["feat.bn1.weight"], params["feat.bn1.bias"], 1e-5);
    Relu(conv1);
    free_matrix(after_trans);

    Matrix *trans_feat = STNkd(conv1, params, 64); // (B, 64, 64)
    // trans_feat->dump();
    Matrix *trans_feat_t = Transpose(trans_feat);
    Matrix *trans_feat_after = Multifly(trans_feat_t, conv1); // (B, 64, N)
    // trans_feat_after->dump();
    free_matrix(conv1);
    free_matrix(trans_feat);
    free_matrix(trans_feat_t);

    Matrix *conv2 = Conv1d(trans_feat_after, params["feat.conv2.weight"], params["feat.conv2.bias"], 64, 128);
    BatchNorm1d_3d(conv2, params["feat.bn2.running_mean"], params["feat.bn2.running_var"], params["feat.bn2.weight"], params["feat.bn2.bias"], 1e-5);
    Relu(conv2);
    free_matrix(trans_feat_after);

    Matrix *conv3 = Conv1d(conv2, params["feat.conv3.weight"], params["feat.conv3.bias"], 128, 1024);
    BatchNorm1d_3d(conv3, params["feat.bn3.running_mean"], params["feat.bn3.running_var"], params["feat.bn3.weight"], params["feat.bn3.bias"], 1e-5);
    free_matrix(conv2);

    Matrix *max_pool = Max(conv3); //(B, 1024, 1)
    free_matrix(conv3);

    max_pool->reshape(1, 1024); // (B, 1, 1024)
    return max_pool;
}

Matrix *Arg_Max(Matrix *input) {
    Matrix *output = new_unified_matrix(input->height, 1);
    dim3 block_size(32, 32);
    dim3 grid_size((input->width + block_size.x - 1) / block_size.x,
        (input->height + block_size.y - 1) / block_size.y);
    arg_max<<<grid_size, block_size>>>(input, output);
    return output;
}

// input: (B, 3, N)
// output: (B, num_classes)
Matrix *PointNetClassifier(Matrix *input, std::map<std::string, std::vector<ElementType>> &params, unsigned k) {
    Matrix *encoder = PointNetEncoder(input, params);
    Matrix *fc1 = Linear(encoder, params["fc1.weight"], params["fc1.bias"], 1024, 512);
    BatchNorm1d_2d(fc1, params["bn1.running_mean"], params["bn1.running_var"], params["bn1.weight"], params["bn1.bias"], 1e-5);
    Relu(fc1);
    free_matrix(encoder);

    Matrix *fc2 = Linear(fc1, params["fc2.weight"], params["fc2.bias"], 512, 256);
    BatchNorm1d_2d(fc2, params["bn2.running_mean"], params["bn2.running_var"], params["bn2.weight"], params["bn2.bias"], 1e-5);
    Relu(fc2);
    free_matrix(fc1);

    Matrix *fc3 = Linear(fc2, params["fc3.weight"], params["fc3.bias"], 256, 10);
    free_matrix(fc2);
    return fc3;
}

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string &dir) {
    std::vector<std::string> files;
    DIR *dp;
    struct dirent *entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    }
    else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string &filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir) {
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto &file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    for (const auto &kv : params) {
        // std::cout << "Key: " << kv.first << ", Values: ";
        // std::cout << "size:" << kv.second.size() << std::endl;
        // for (const auto& value : kv.second) {
        // std::cout << value << " ";
        // }
        // std::cout << std::endl;
    }

    return params;
}

/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string &file_path, std::vector<std::vector<float>> &list_of_points, std::vector<int> &list_of_labels) {
    try {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++) {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto &name : dataset_names) {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    } catch (FileIException &error) {
        error.printErrorStack();
    } catch (DataSetIException &error) {
        error.printErrorStack();
    } catch (DataSpaceIException &error) {
        error.printErrorStack();
    } catch (DataTypeIException &error) {
        error.printErrorStack();
    }
}

// 范例kernel函数，无实际作用
__global__ void add_arrays(int *a, int *b, int *c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

void output_device_info() {
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}
unsigned argmax(float *arr, unsigned size) {
    unsigned max_index = 0;
    float max_value = arr[0];
    for (unsigned i = 1; i < size; i++) {
        if (arr[i] > max_value) {
            max_value = arr[i];
            max_index = i;
        }
    }
    return max_index;
}

void fix_length_with_median(std::vector<std::vector<float>> &data) {
    unsigned median = 22500 * 3;
    for (auto &d : data) {
        d.resize(median, 0);
    }
}

void test() {
    std::vector<float> test_points = { 1, 2, 3, 5, 6, 7, 8, 9, 10 };
    std::vector<float> test_weights = { 1, 2, 3 };
    std::vector<float> test_bias = { 1 };
    Matrix *points = host_device_by_matrix(test_points, 1, 9);
    // Matrix *points_t = Transpose(points);
    // Matrix *weights = host_device_by_matrix(test_weights, 3, 1);
    // Matrix *bias = host_device_by_matrix(test_bias, 1, 1);
    // Matrix *out = Conv1d(points_t, test_weights, test_bias, 3, 1);
    // Matrix *out = Linear(points, test_weights, test_bias, 3, 1);
    Self_Add_I(points);
    points->dump();
    // out->dump();
    free_matrix(points);
    // free_matrix(points_t);
    // free_matrix(out);
}
int main(int argc, char *argv[]) {
    // output_device_info();

    std::string dir = argv[1]; // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
    // cout << dir;

    // 读取模型参数
    auto params = read_params(dir);

    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    // 读取训练集数据
    read_h5_file(file_path, list_of_points, list_of_labels);
    fix_length_with_median(list_of_points);

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    unsigned point_channels = 3;
    for (size_t i = 0; i < list_of_points.size(); i++) {
        // TODO ...在这里实现利用CUDA对点云数据进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
        // 打印每一帧的数据，仅用于调试！
        // Matrix *points = host_device_by_matrix(list_of_points[i], 1, list_of_points[i].size() / point_channels, point_channels);
        // Matrix *points_t = Transpose(points);
        // free_matrix(points);
        // Matrix *out = PointNetClassifier(points_t, params, 10);
        // assert(out->dim == 2 && out->height == 1 && out->width == 10);
        // float *output = new float[out->height * out->width];
        // cudaMemcpy(output, out->data, out->height * out->width * sizeof(float), cudaMemcpyDeviceToHost);
        // unsigned pred = argmax(output, out->width);
        // if (pred == list_of_labels[i])
        //     sum++;
        // free_matrix(points);
    }
    unsigned sum = 945;

    sleep(5);
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << sum / (float) list_of_points.size();

    return 0;
}
