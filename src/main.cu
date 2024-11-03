// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include "matrix.cuh"
#include "models.cuh"
#include "utils.cuh"
#include <chrono>
#include <dirent.h>
#include <hdf5/serial/H5Cpp.h>
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
        std::cout << "Key: " << kv.first << ", Values: ";
        std::cout << "size:" << kv.second.size() << std::endl;
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
    std::vector<float> test_points = { 1, 2, 3, 5, 6, 7, 8, 9, 10, 1, 2, 3, 5, 6, 7, 8, 9, 10 };
    std::vector<float> test_weights = { 1, 2, 3, 1, 2, 3 };
    std::vector<float> test_bias = { 1, 2 };
    Matrix *points = host_device_by_matrix(test_points, 2, 3, 3);
    // Matrix *points_t = Transpose(points);
    // Matrix *weights = host_device_by_matrix(test_weights, 3, 2);
    // Matrix *bias = host_device_by_matrix(test_bias, 2, 1);
    // Matrix *out = Conv1d(points_t, test_weights, test_bias, 3, 2);
    // Matrix *out = Max(points);
    Matrix *points_t = Transpose(points);
    // Matrix *out = Linear(points, test_weights, test_bias, 3, 2);
    // Matrix *out = Multifly_With_T(points_t, weights);
    // Self_Add_I(points);
    // weights->dump();
    points->dump();
    points_t->dump();
    // out->dump();
    free_matrix(points);
    // free_matrix(points_t);
    // free_matrix(out);
}

std::vector<std::vector<float>> scatter_to_batch(std::vector<std::vector<float>> &data, unsigned batch_size) {
    std::vector<std::vector<float>> batch_data;
    for (size_t i = 0; i < data.size(); i += batch_size) {
        std::vector<float> batch;
        for (size_t j = 0; j < batch_size; j++) {
            if (i + j < data.size()) {
                batch.insert(batch.end(), data[i + j].begin(), data[i + j].end());
            }
            else {
                batch.insert(batch.end(), data[i].size(), 0);
            }
        }
        batch_data.push_back(batch);
    }
    return batch_data;
}
int main(int argc, char *argv[]) {
    // output_device_info();

    std::string dir = argv[1]; // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
    // cout << dir;

    // 读取模型参数
    std::string model_path = dir + "/params";
    auto params = read_params(model_path);

    std::string file_path = dir + "/data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    // 读取训练集数据
    read_h5_file(file_path, list_of_points, list_of_labels);
    fix_length_with_median(list_of_points);
    test();
    unsigned point_channels = 3;
    unsigned sum = 0;
    std::vector<std::vector<float>> list_of_points_batch = scatter_to_batch(list_of_points, BLOCK_SIZE);
    // return;
    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 1; i < list_of_points_batch.size(); i++) {
        // TODO ...在这里实现利用CUDA对点云数据进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
        // 打印每一帧的数据，仅用于调试！
        // std::cout << list_of_points_batch[i].size() << " " << BLOCK_SIZE * 22500 * 3 << std::endl;
        // std::cout << list_of_points_batch[i][22500 * 3 + 5] << std::endl;
        Matrix *points = host_device_by_matrix(list_of_points_batch[i], BLOCK_SIZE, list_of_points_batch[i].size() / point_channels / BLOCK_SIZE, point_channels);
        // points->dump();
        Matrix *points_t = Transpose(points);
        // points_t->dump();

        free_matrix(points);
        Matrix *out = PointNetClassifier(points_t, params, 10);
        assert(out->dim == 2 && out->height == BLOCK_SIZE && out->width == 10);
        float *output = new float[out->height * out->width];
        cudaMemcpy(output, out->data, out->height * out->width * sizeof(float), cudaMemcpyDeviceToHost);
        for (int j = 0; j < BLOCK_SIZE; j++) {
            if (i*BLOCK_SIZE + j >= list_of_points.size()) break;
            unsigned pred = argmax(output + j * out->width, out->width);
            if (list_of_labels[i * BLOCK_SIZE + j] == pred) {
                std::cout << i * BLOCK_SIZE + j << ": Predicted label: " << pred << " True label" << list_of_labels[i * BLOCK_SIZE + j] << " success" << std::endl;
                sum++;
            }
            else
                std::cout << i * BLOCK_SIZE + j << ": Predicted label: " << pred << " True label" << list_of_labels[i * BLOCK_SIZE + j] << " failed" << std::endl;
        }
        free_matrix(points);
        // break;
        // if (i == 1) break;
    }
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << sum / (float) list_of_points.size();

    return 0;
}
