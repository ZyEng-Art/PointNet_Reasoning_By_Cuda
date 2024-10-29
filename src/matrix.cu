#include "matrix.cuh"

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