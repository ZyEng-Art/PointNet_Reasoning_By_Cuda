#ifndef __MATRIX__
#define __MATRIX__

#include "utils.cuh"
class Matrix {
private:
public:
    ElementType *data;
    unsigned height;
    unsigned width;

    __device__ ElementType get_element(unsigned row, unsigned col) {
        return this->data[row * this->width + col];
    }
    __device__ void set_element(unsigned row, unsigned col, ElementType element) {
        this->data[row * this->width + col] = element;
    }
    Matrix(ElementType *data, unsigned height, unsigned width) {
        this->data = data;
        this->height = height;
        this->width = width;
    }
    ~Matrix() {
        if (data)
            cudaFree(data);
    }
};

Matrix *new_unified_matrix(unsigned height, unsigned width);
Matrix *host_device_by_matrix(std::vector<ElementType> &host, unsigned height, unsigned width);
void free_matrix(Matrix *matrix);
#endif