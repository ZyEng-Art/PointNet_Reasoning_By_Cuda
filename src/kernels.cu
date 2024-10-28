// #include "kernels.cuh"
// template <typename T>
// __global__ void conv1d_1kernl(Matrix<T> *input, Matrix<T> *output, Matrix<T> *weight, Matrix<T> *bias) {
//     // input:: length x in_channels
//     // output:: length x out_channels
//     // kernel:: in_channels x out_channels
//     // 如果kernel 很小，可以放入constant memory?
//     // 不使用shared memory
//     unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
//     unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row >= output->height || col >= output->width) {
//         return;
//     }
//     T sum = 0;
//     for (unsigned i = 0; i < input->width /*weight->height*/; i++) {
//         sum += get_element(input, row, i) * get_element(weight, i, col) + get_element(bias, 0, col);
//     }
//     set_element(output, row, col, sum);
// }

// template<typename T>
// __global__ void add(Matrix<T> *a, Matrix<T> *b, Matrix<T> *c) {
//     unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
//     unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row >= c->height || col >= c->width) {
//         return;
//     }
//     set_element(c, row, col, get_element(a, row, col) + get_element(b, row, col));
// }

