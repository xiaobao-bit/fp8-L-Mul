//
//  func_test.cpp
//  l_mul
//
//  Created by Han Bao on 24/02/2025.
//

#include "func_test.hpp"

std::vector<std::vector<float>> generate_random_fp32_matrix(int rows, int cols, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = dis(gen);
        }
    }
    
    return mat;
}

void print_matrix(const std::vector<std::vector<float>>& mat, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (const auto& row : mat) {
        for (float val : row) {
            std::cout << std::fixed << std::setprecision(4) << val << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::vector<std::vector<float>> fp32_matmul(const std::vector<std::vector<float>>& mat_x,
                                            const std::vector<std::vector<float>>& mat_y) {
    int rows_x = static_cast<int> (mat_x.size());
    int cols_x = static_cast<int> (mat_x[0].size());
    int cols_y = static_cast<int> (mat_y[0].size());

    std::vector<std::vector<float>> res(rows_x, std::vector<float>(cols_y, 0.0f));

    for (int i = 0; i < rows_x; ++i) {
        for (int j = 0; j < cols_y; ++j) {
            for (int k = 0; k < cols_x; ++k) {
                res[i][j] += mat_x[i][k] * mat_y[k][j];
            }
        }
    }

    return res;
}

int main() {
    int rows_x = 3, cols_x = 2;
    int rows_y = 2, cols_y = 3;
    int num_exp = 4;

    auto fp32_mat_x = generate_random_fp32_matrix(rows_x, cols_x);
    auto fp32_mat_y = generate_random_fp32_matrix(rows_y, cols_y);

    print_matrix(fp32_mat_x, "X (FP32)");
    print_matrix(fp32_mat_y, "Y (FP32)");

    try {
        auto res_mat = operands::lmul_matmul(fp32_mat_x, fp32_mat_y, num_exp);
        print_matrix(res_mat, "Result (LMUL MatMul)");
        
        auto res_mat_fp32 = fp32_matmul(fp32_mat_x, fp32_mat_y);
        print_matrix(res_mat_fp32, "Result (Exact FP32 MatMul)");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
