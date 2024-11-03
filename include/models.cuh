#ifndef __MODELS__
#define __MODELS__

#include "matrix.cuh"
#include "utils.cuh"

Matrix *Conv1d(Matrix *input, std::vector<ElementType> &weight, std::vector<ElementType> &bias, unsigned in_channels, unsigned out_channels);
void Relu(Matrix *input);
Matrix *Max(Matrix *input);
void BatchNorm1d(Matrix *input, std::vector<ElementType> &running_mean, std::vector<ElementType> &running_var, std::vector<ElementType> &gamma, std::vector<ElementType> &beta, ElementType eps);
void Self_Add_I(Matrix *input);
Matrix *Transpose(Matrix *input);
Matrix *Linear(Matrix *input, std::vector<ElementType> &weight, std::vector<ElementType> &bias, unsigned in_features, unsigned out_features) ;
Matrix *Multifly_With_T(Matrix *input, Matrix *weight);

Matrix *STN3d(Matrix *input, std::map<std::string, std::vector<ElementType>> &params, ElementType eps=1e-5);
Matrix *STNkd(Matrix *input, std::map<std::string, std::vector<ElementType>> &params);
Matrix *PointNetEncoder(Matrix *input, std::map<std::string, std::vector<ElementType>> &params);
Matrix *PointNetClassifier(Matrix *input, std::map<std::string, std::vector<ElementType>> &params, unsigned num_classes);

#endif