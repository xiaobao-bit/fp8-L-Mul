//
//  operator.hpp
//  l_mul
//
//  Created by Han Bao on 24/02/2025.
//

#ifndef operator_hpp
#define operator_hpp

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>
#include <utility>
#include "format_conversion.hpp"

class operands {
private:
    static std::tuple<int, float, float, float> fp8_decode(uint8_t fp8_elem, int num_exp);
    static std::tuple<int, float, float> fp8_res_tpl(uint8_t fp8_elem, int num_exp);
    static float fp8_res_val(uint8_t fp8_elem, int num_exp);
    static float lmul_single(uint8_t fp32_uint8_x, uint8_t fp32_uint8_y, int num_exp);
    
public:
    static std::vector<std::vector<float>> lmul_matmul(std::vector<std::vector<float>> fp32_mat_x, std::vector<std::vector<float>> fp32_mat_y, int num_exp);
};

#endif /* operator_hpp */
