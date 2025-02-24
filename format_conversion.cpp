//
//  fp32_to_fp8.cpp
//  l_mul
//
//  Created by Han Bao on 24/02/2025.
//

#include "format_conversion.hpp"


uint8_t formatConversion::fp32_downcast(float input_fp32, int num_exp) {
        const int source_m_nbits = 23;  // FP32 mantissa bits
        const int source_e_nbits = 8;   // FP32 exponent bits
        const int target_e_nbits = num_exp;
        const int target_m_nbits = 7 - num_exp;

        if (target_e_nbits + target_m_nbits + 1 > 8) {
            throw std::invalid_argument("Mantissa is too large for an 8-bit float");
        }

        // IEEE 754 bit representation
        union {
            float f;
            uint32_t i;
        } u;
        u.f = input_fp32;

        // Extract sign, exponent, and mantissa
        uint8_t sign = (u.i >> 31) & 0x1;
        uint32_t fp32_exp = (u.i >> source_m_nbits) & 0xFF;
        uint32_t fp32_mantissa = u.i & ((1u << source_m_nbits) - 1);

        // Compute FP8 bias
        int fp32_bias = 127;
        int fp8_bias = (1 << (target_e_nbits - 1)) - 1;

        // Compute new exponent
        int fp8_exp = fp32_exp - fp32_bias + fp8_bias;
        bool is_subnormal = false;

        // Handle subnormal numbers
        if (fp8_exp < 1) {
            is_subnormal = true;
            fp8_exp = 0;
            int shift = 1 - (fp32_exp - fp32_bias);
            fp32_mantissa = (0x800000 | fp32_mantissa) >> shift;
        }
        // Handle exponent overflow (set to infinity)
        else if (fp8_exp >= (1 << target_e_nbits)) {
            fp8_exp = (1 << target_e_nbits) - 1;
            fp32_mantissa = (1 << target_m_nbits) - 1;
        }
        
        fp8_exp = std::max(1, std::min(fp8_exp, (1 << target_e_nbits) - 1));

        // Truncate mantissa
        int mantissa_shift = source_m_nbits - target_m_nbits;
        uint8_t truncated_mantissa = fp32_mantissa >> mantissa_shift;

        // Compute probability for stochastic rounding
        uint32_t remainder = fp32_mantissa & ((1 << mantissa_shift) - 1);
        float probability = static_cast<float>(remainder) / (1 << mantissa_shift);

        // Random rounding decision
        static std::mt19937 gen{std::random_device{}()};
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(gen) < probability && truncated_mantissa < (1 << target_m_nbits) - 1) {
            truncated_mantissa += 1;
        }
        if (truncated_mantissa >= (1 << target_m_nbits)) {
            truncated_mantissa = 0;
            fp8_exp += 1;
            if (fp8_exp >= (1 << target_e_nbits)) {
                fp8_exp = (1 << target_e_nbits) - 1;
                truncated_mantissa = (1 << target_m_nbits) - 1;
            }
        }
        // Compose final FP8 representation
        uint8_t fp8 = (sign << 7) | (fp8_exp << target_m_nbits) | truncated_mantissa;
        return fp8;
    }

std::vector<std::vector<uint8_t>> formatConversion::fp32_mat_downcast(std::vector<std::vector<float>> fp32_mat, int num_exp) {
    if (fp32_mat.empty()) {
        throw std::invalid_argument("Empty input matrix");
    }
    
    std::vector<std::vector<uint8_t>> fp8_mat(fp32_mat.size(), std::vector<uint8_t>(fp32_mat[0].size()));
    
    // #pragma omp parallel for collapse(2) num_threads(8)
    for (size_t i = 0; i < fp32_mat.size(); i++) {
            for (size_t j = 0; j < fp32_mat[i].size(); j++) {
                fp8_mat[i][j] = formatConversion::fp32_downcast(fp32_mat[i][j], num_exp);
            }
        }
    return fp8_mat;
}
