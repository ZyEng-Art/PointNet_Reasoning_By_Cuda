
#include "kernels.cuh"
#include "models.cuh"

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
    dim3 grid_size(
        (output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y,
        output->batch);
    conv1d_1kernel_shared_mem<<<grid_size, block_size>>>(input, output, weights, biases);
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
    linear_shared_mem<<<grid_size, block_size>>>(input, output, weights, biases);
    free_matrix(weights);
    free_matrix(biases);
    return output;
}

void Relu(Matrix *input) {
    if (input->dim == 2) {
        dim3 block_size(32, 32);
        dim3 grid_size((input->width + block_size.x - 1) / block_size.x,
            (input->height + block_size.y - 1) / block_size.y);
        relu2<<<grid_size, block_size>>>(input);
    }
    else {
        dim3 block_size(32, 32);
        dim3 grid_size((input->width + block_size.x - 1) / block_size.x,
            (input->height + block_size.y - 1) / block_size.y,
            input->batch);
        relu3<<<grid_size, block_size>>>(input);
    }
}

Matrix *Transpose(Matrix *input) {
    assert(input->dim == 3);
    Matrix *output = new_unified_matrix(input->batch, input->width, input->height);
    dim3 block_size(32, 32);
    dim3 grid_size(
        (output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y,
        output->batch);
    transpose<<<grid_size, block_size>>>(input, output);
    return output;
}

Matrix *Max(Matrix *input) {
    assert(input->dim == 3);
    Matrix *output = new_unified_matrix(input->batch, input->height);
    dim3 block_size(32, 32);
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
    dim3 grid_size(
        (input->width + block_size.x - 1) / block_size.x,
        (input->height + block_size.y - 1) / block_size.y,
        input->batch);
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

Matrix *STN3d(Matrix *input, std::map<std::string, std::vector<ElementType>> &params, ElementType eps) {
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
    // conv2->dump();
    free_matrix(conv1);
    Matrix *conv3 = Conv1d(conv2, params["feat.stn.conv3.weight"], params["feat.stn.conv3.bias"], 128, 1024);
    BatchNorm1d_3d(conv3, params["feat.stn.bn3.running_mean"], params["feat.stn.bn3.running_var"], params["feat.stn.bn3.weight"], params["feat.stn.bn3.bias"], eps);
    Relu(conv3);
    free_matrix(conv2);
    // conv3->dump();
    Matrix *max_pool = Max(conv3); //(B, 1024, 1)
    free_matrix(conv3);

    // max_pool->dump();
    max_pool->reshape(input->batch, 1024); // (B, 1024)
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

    max_pool->reshape(input->batch, 1024);
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

// input d x n, weight d x m, output m x n (input^T * weight)^T=weight^T * input
Matrix *Multifly_With_T(Matrix *input, Matrix *weight) {
    assert(input->dim == 3);
    assert(input->height == weight->height);
    Matrix *output = new_unified_matrix(input->batch, weight->width, input->width);
    dim3 block_size(32, 32);
    dim3 grid_size(
        (output->width + block_size.x - 1) / block_size.x,
        (output->height + block_size.y - 1) / block_size.y,
        output->batch);
    multifly_with_t_shared_mem<<<grid_size, block_size>>>(input, output, weight);
    return output;
}

// input: (B, 3, N)
Matrix *PointNetEncoder(Matrix *input, std::map<std::string, std::vector<ElementType>> &params) {
    Matrix *trans = STN3d(input, params); // (B, 3, 3)
    // trans->dump();
    Matrix *after_trans = Multifly_With_T(input, trans); // (B, 3, N)
    // after_trans->dump();
    free_matrix(trans);

    Matrix *conv1 = Conv1d(after_trans, params["feat.conv1.weight"], params["feat.conv1.bias"], 3, 64); //(B, 64, N)
    BatchNorm1d_3d(conv1, params["feat.bn1.running_mean"], params["feat.bn1.running_var"], params["feat.bn1.weight"], params["feat.bn1.bias"], 1e-5);
    Relu(conv1);
    free_matrix(after_trans);

    Matrix *trans_feat = STNkd(conv1, params, 64); // (B, 64, 64)
    // trans_feat->dump();
    Matrix *trans_feat_after = Multifly_With_T(conv1, trans_feat); // (B, 64, N)
    // trans_feat_after->dump();
    free_matrix(conv1);
    free_matrix(trans_feat);

    Matrix *conv2 = Conv1d(trans_feat_after, params["feat.conv2.weight"], params["feat.conv2.bias"], 64, 128);
    BatchNorm1d_3d(conv2, params["feat.bn2.running_mean"], params["feat.bn2.running_var"], params["feat.bn2.weight"], params["feat.bn2.bias"], 1e-5);
    Relu(conv2);
    free_matrix(trans_feat_after);

    Matrix *conv3 = Conv1d(conv2, params["feat.conv3.weight"], params["feat.conv3.bias"], 128, 1024);
    BatchNorm1d_3d(conv3, params["feat.bn3.running_mean"], params["feat.bn3.running_var"], params["feat.bn3.weight"], params["feat.bn3.bias"], 1e-5);
    free_matrix(conv2);

    Matrix *max_pool = Max(conv3); //(B, 1024, 1)
    free_matrix(conv3);

    max_pool->reshape(input->batch, 1024); // (B, 1024)
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
