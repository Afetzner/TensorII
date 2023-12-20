//
// Created by Amy Fetzner on 12/18/2023.
//

#include "Catch2/catch_test_macros.hpp"
#include "TensorII/Tensor.h"
#include "TensorII/TensorView.h"

using namespace TensorII::Core;

TEST_CASE("TensorView Initialization", "[TensorView]"){
    Tensor tensor = toTensor({{1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9}});
    TensorView tensorView = tensor;
    tensorView[{}];
    tensorView[1];
    tensorView[{1, 2}];
    tensorView[{1, 2, 3}];
}