
#include "kernels.cuh"
#include "models.cuh"

// input 在设备内存
Matrix *Conv1d(Matrix *input, std::vector<ElementType> &weight, std::vector<ElementType> &bias, unsigned in_channels, unsigned out_channels) {
    std::cout << weight.size() << " " << in_channels << " " << out_channels << std::endl;
    assert(weight.size() == in_channels * out_channels);
    assert(bias.size() == out_channels);
    assert(input->width == in_channels);
    Matrix *weights = host_device_by_matrix(weight, in_channels, out_channels);
    Matrix *biases = host_device_by_matrix(bias, 1, out_channels);
    Matrix *output = new_unified_matrix(input->height, out_channels);

    dim3 block_size(32, 32);
    dim3 grid_size((output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y);
    conv1d_1kernel<<<grid_size, block_size>>>(input, output, weights, biases);
    return output;
}

Matrix *STN3d(Matrix *input, std::map<std::string, std::vector<ElementType>> &params) {
    Matrix *conv1 = Conv1d(input, params["feat.stn.conv1.weight"], params["feat.stn.conv1.bias"], 3, 64);
    Matrix *conv2 = Conv1d(conv1, params["feat.stn.conv2.weight"], params["feat.stn.conv2.bias"], 64, 128);
    free_matrix(conv1);
    Matrix *conv3 = Conv1d(conv2, params["feat.stn.conv3.weight"], params["feat.stn.conv3.bias"], 128, 1024);
    free_matrix(conv2);

    return conv3;
}

Matrix *STNkd(Matrix *input, std::map<std::string, std::vector<ElementType>> &params) {
    Matrix *conv1 = Conv1d(input, params["feat.stn.conv1_weight"], params["feat.stn.conv1_bias"], 3, 64);
    Matrix *conv2 = Conv1d(conv1, params["feat.stn.conv2_weight"], params["feat.stn.conv2_bias"], 64, 128);
    free_matrix(conv1);
    Matrix *conv3 = Conv1d(conv2, params["feat.stn.conv3_weight"], params["feat.stn.conv3_bias"], 128, 1024);
    free_matrix(conv2);

    return conv3;
}

Matrix *PointNetEncoder(Matrix *input, std::map<std::string, std::vector<ElementType>> &params) {
    Matrix *stn = STN3d(input, params);
    return stn;
}

Matrix *PointNetClassifier(Matrix *input, std::map<std::string, std::vector<ElementType>> &params) {
    Matrix *encoder = PointNetEncoder(input, params);
    // Matrix<T> *conv1 = Conv1d<T>(encoder, params["conv1_weight"], params["conv1_bias"], 1024, 512);
    // Matrix<T> *conv2 = Conv1d<T>(conv1, params["conv2_weight"], params["conv2_bias"], 512, 256);
    // Matrix<T> *conv3 = Conv1d<T>(conv2, params["conv3_weight"], params["conv3_bias"], 256, 128);
    // Matrix<T> *conv4 = Conv1d<T>(conv3, params["conv4_weight"], params["conv4_bias"], 128, 40);
    // free_matrix(conv1);
    // free_matrix(conv2);
    // free_matrix(conv3);
    return encoder;
}
