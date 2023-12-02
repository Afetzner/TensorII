//
// Created by Amy Fetzner on 12/2/2023.
//

#include "Catch2/catch_test_macros.hpp"
#include "TensorII/Tensor.h"

using namespace TensorII::Core;

TEST_CASE("Tensor Indexing", "[Tensor][Index]"){
    Tensor t = Tensor<int, Shape<5, 3, 2>> ({{{000, 001}, {010, 011}, {020, 021}},
                                             {{100, 101}, {110, 111}, {120, 121}},
                                             {{200, 201}, {210, 211}, {220, 221}},
                                             {{300, 301}, {310, 311}, {320, 321}},
                                             {{400, 401}, {410, 411}, {420, 421}}});
    auto* ptr = t[{1, 2, 3}];
    CHECK(*ptr == 0);
}