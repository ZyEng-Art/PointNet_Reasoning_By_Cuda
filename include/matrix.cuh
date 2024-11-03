#ifndef __MATRIX__
#define __MATRIX__

#include "utils.cuh"
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
            float *output = new float[height * width];
            cudaMemcpy(output, data, height * width * sizeof(ElementType), cudaMemcpyDeviceToHost);
            for (int i = 0; i < height && i < 5; i++) {
                std::cout << "row " << i << ": ";
                for (int j = 0; j < width && j < 5; j++) {
                    std::cout << output[i * width + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "finish" << std::endl;
            delete output;
            return;
        }
        std::cout << "size:(" << batch << "," << height << "," << width << ")" << std::endl;
        float *output = new float[batch * height * width];
        cudaMemcpy(output, data, batch * height * width * sizeof(ElementType), cudaMemcpyDeviceToHost);
        for (int k = 0; k < batch && k < 5; k++) {
            std::cout << "batch " << k << std::endl;
            for (int i = 0; i < height && i < 5; i++) {
                std::cout << "     row " << i << ": ";
                for (int j = 0; j < width && j < 5; j++) {
                    std::cout << output[k * width * height + i * width + j] << " ";
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

Matrix *new_unified_matrix(unsigned height, unsigned width);
Matrix *new_unified_matrix(unsigned batch, unsigned height, unsigned width);
Matrix *host_device_by_matrix(std::vector<ElementType> &host, unsigned height, unsigned width);
Matrix *host_device_by_matrix(std::vector<ElementType> &host, unsigned batch, unsigned height, unsigned width);
void free_matrix(Matrix *matrix);
#endif