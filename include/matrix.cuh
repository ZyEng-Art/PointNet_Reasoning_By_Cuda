#ifndef __MATRIX__
#define __MATRIX__

#include "utils.cuh"
template <typename T>
class Matrix {
private:
public:
    T *data;
    unsigned height;
    unsigned width;

    __device__ T get_element(unsigned row, unsigned col) {
        return this->data[row * this->width + col];
    }
    __device__ void set_element(unsigned row, unsigned col, T element) {
        this->data[row * this->width + col] = element;
    }
    Matrix(T *data, unsigned height, unsigned width) {
        this->data = data;
        this->height = height;
        this->width = width;
    }
    ~Matrix() {
        if (data)
            cudaFree(data);
    }
};

template <typename T>
Matrix<T> *new_unified_matrix(unsigned height, unsigned width) {
    T *device_array;
    cudaMalloc(&device_array, height * width * sizeof(T));
    Matrix<T> *unified_matrix;
    cudaMallocManaged(&unified_matrix, sizeof(Matrix<T>));
    unified_matrix->height = height;
    unified_matrix->width = width;
    unified_matrix->data = device_array;
    return unified_matrix;
}

template <typename T>
Matrix<T> *host_device_by_matrix(std::vector<T> &host, unsigned height, unsigned width) {
    assert(host.size() == height * width);
    Matrix<T> *device_matrix = new_unified_matrix<T>(height, width);
    cudaMemcpy(device_matrix->data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device_matrix;
}

template <typename T>
void free_matrix(Matrix<T> *matrix) {
    matrix->~Matrix();
    cudaFree(matrix);
}

template <typename T>
Matrix<T> *new_unified_matrix(unsigned height, unsigned width);
template <typename T>
Matrix<T> *host_device_by_matrix(std::vector<T> &host, unsigned height, unsigned width);
template <typename T>
void free_matrix(Matrix<T> *matrix);
#endif