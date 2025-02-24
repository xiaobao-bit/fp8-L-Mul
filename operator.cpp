//
//  operator.cpp
//  l_mul
//
//  Created by Han Bao on 24/02/2025.
//

#include "operator.hpp"

std::tuple<int, float, float, float> operands::fp8_decode(uint8_t fp8_elem, int num_exp) {
    int num_man = 7 - num_exp;

    static const std::array<std::pair<uint8_t, uint8_t>, 6> masks = {{
        {0x40, 0x3F},  // num_exp = 1
        {0x60, 0x1F},  // num_exp = 2
        {0x70, 0x0F},  // num_exp = 3
        {0x78, 0x07},  // num_exp = 4
        {0x7C, 0x03},  // num_exp = 5
        {0x7E, 0x01}   // num_exp = 6
    }};

    if (num_exp < 1 || num_exp > static_cast<int>(masks.size())) {
        throw std::invalid_argument("Invalid number of exponent");
    }

    auto [exp_mask, man_mask] = masks[num_exp - 1];

    int sign = (fp8_elem & 0x80) >> 7;
    int exp = (fp8_elem & exp_mask) >> num_man;
    int man = fp8_elem & man_mask;

    int bias = (1 << (num_exp - 1)) - 1;
    float E = (exp == 0) ? 0 : (exp - bias);
    float f = man / static_cast<float>(1 << num_man);
    float M = (exp == 0) ? 0 : (1.0f + f);

    return {sign, E, f, M};
}

float operands::fp8_res_val(uint8_t fp8_elem, int num_exp) {
    auto tpl_fp8_res_val = operands::fp8_decode(fp8_elem, num_exp);
    int sign = std::get<0> (tpl_fp8_res_val);
    float E = std::get<1> (tpl_fp8_res_val);
    float M = std::get<3> (tpl_fp8_res_val);
    return (sign ? -1.0f : 1.0f) * std::powf(2.0f, E) * M;
}

std::tuple<int, float, float> operands::fp8_res_tpl(uint8_t fp8_elem, int num_exp) {
    auto [sign, E, f, M] = operands::fp8_decode(fp8_elem, num_exp);
    return std::make_tuple(sign, E, f);
}

float operands::lmul_single(uint8_t fp32_uint8_x, uint8_t fp32_uint8_y, int num_exp) {
    int num_man = 7 - num_exp;
    
    std::tuple<int, float, float> res_x = operands::fp8_res_tpl(fp32_uint8_x, num_exp);
    std::tuple<int, float, float> res_y = operands::fp8_res_tpl(fp32_uint8_y, num_exp);
    
    int lm;
    if (num_man <= 3) lm = num_man;
    else if (num_man == 4) lm = 3;
    else lm = 4;
    
    int sign = (std::get<0>(res_x) == std::get<0>(res_y)) ? 1 : -1;
    
    float Ex = std::get<1>(res_x);
    float Ey = std::get<1>(res_y);
    float fx = std::get<2>(res_x);
    float fy = std::get<2>(res_y);
    
    // #pragma omp parallel for collapse(2) num_threads(8)
    float res = sign * (1.0f + fx + fy + std::powf(2.0f, -lm)) * std::powf(2.0f, Ex + Ey);
    // float res = sign * (1.0f + fx) * (1.0f + fy) * std::powf(2.0f, Ex + Ey);
    return res;
}

std::vector<std::vector<float>> operands::lmul_matmul(std::vector<std::vector<float>> fp32_mat_x, std::vector<std::vector<float>> fp32_mat_y, int num_exp) {
    
    std::vector<std::vector<uint8_t>> fp8_mat_x = formatConversion::fp32_mat_downcast(fp32_mat_x, num_exp);
    std::vector<std::vector<uint8_t>> fp8_mat_y = formatConversion::fp32_mat_downcast(fp32_mat_y, num_exp);
    
    if (fp8_mat_x.empty() || fp8_mat_y.empty()) {
        throw std::invalid_argument("Empty input matrix");
    }

    int rows_x = static_cast<int>(fp8_mat_x.size());
    int cols_x = static_cast<int>(fp8_mat_x[0].size());
    int rows_y = static_cast<int>(fp8_mat_y.size());
    int cols_y = static_cast<int>(fp8_mat_y[0].size());
    
    if (cols_x != rows_y) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }

    std::vector<std::vector<float>> res_mat(rows_x, std::vector<float>(cols_y, 0.0f));
    
    std::cout << "FP8 Matrix X:\n";
        for (const auto& row : fp8_mat_x) {
            for (uint8_t val : row) {
                std::cout << fp8_res_val(val, num_exp) << "\t";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nFP8 Matrix Y:\n";
        for (const auto& row : fp8_mat_y) {
            for (uint8_t val : row) {
                std::cout << fp8_res_val(val, num_exp) << "\t";
            }
            std::cout << std::endl;
        }
    std::cout << std::endl;
    
    // #pragma omp parallel for collapse(2) num_threads(8)
    for (int i = 0; i < rows_x; i++) {
        for (int j = 0; j < cols_y; j++) {
            float sum = 0.0f;
            // #pragma omp simd reduction(+:sum)
            for (int k = 0; k < cols_x; k++) {
                sum += operands::lmul_single(fp8_mat_x[i][k], fp8_mat_y[k][j], num_exp);
            }
            res_mat[i][j] = sum;
        }
    }
    
    return res_mat;
}
