//
// Created by Amy Fetzner on 12/18/2023.
//

#include "Catch2/catch_test_macros.hpp"
#include "TensorII/Tensor.h"
#include "TensorII/TensorView.h"

using namespace TensorII::Core;

TEST_CASE("TensorView Initialization", "[TensorView]"){
    constexpr Tensor tensor = toTensor({{1, 2, 3},
                                        {4, 5, 6},
                                        {7, 8, 9}});
    constexpr TensorView tensorView = tensor;
    SECTION("Empty index"){
        constexpr auto tv = tensorView[{}];
        STATIC_CHECK(tv.shape() == Shape{3});
    }
    SECTION("Single index"){
        constexpr auto tv = tensorView[1];
    }
    SECTION("Start-stop index") {
        constexpr auto tv = tensorView[{1, 2}];
    }
}