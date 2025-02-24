//
//  fp32_to_fp8.hpp
//  l_mul
//
//  Created by Han Bao on 24/02/2025.
//

#ifndef fp32_to_fp8_hpp
#define fp32_to_fp8_hpp

#include <stdio.h>
#include <random>
#include <cmath>
#include <vector>
#include <stdexcept>

class formatConversion {
private:
    static uint8_t fp32_downcast(float input_fp32, int num_exp);
public:
    static std::vector<std::vector<uint8_t>> fp32_mat_downcast(const std::vector<std::vector<float>> fp32_mat, int num_exp);
};

#endif /* fp32_to_fp8_hpp */
