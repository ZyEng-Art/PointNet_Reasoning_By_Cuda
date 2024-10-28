#ifndef __MODELS__
#define __MODELS__

#include "matrix.cuh"
#include "utils.cuh"
Matrix *STN3d(Matrix *input, std::map<std::string, std::vector<ElementType>> &params);
Matrix *STNkd(Matrix *input, std::map<std::string, std::vector<ElementType>> &params);
Matrix *PointNetEncoder(Matrix *input, std::map<std::string, std::vector<ElementType>> &params);
Matrix *PointNetClassifier(Matrix *input, std::map<std::string, std::vector<ElementType>> &params);

#endif