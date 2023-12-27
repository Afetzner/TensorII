//
// Created by Amy Fetzner on 12/18/2023.
//

#include "Catch2/catch_test_macros.hpp"
#include "TensorII/Tensor.h"
#include "TensorII/TensorView.h"

using namespace TensorII::Core;

TEST_CASE("TensorView Initialization", "[TensorView]"){
    Tensor tensor = toTensor<int>({{1,  2,  3,  4},
                                   {5,  6,  7,  8},
                                   {9, 10, 11, 12}});
    TensorView tensorView = tensor;
    SECTION("Empty index"){
        auto tv = tensorView[Empty{}];
        CHECK(tv.shape() == Shape{3, 4});
    }
    SECTION("Single index"){
        auto tv = tensorView[1];
        CHECK(tv.shape() == Shape{4});
    }
    SECTION("Start-stop index") {
//        auto tv = tensorView[{1, 2, 3}];
        static constexpr Triple triple = Triple{1, 2, 3};
        auto tv = tensorView.slice<triple>();
        CHECK(tv.shape() == Shape{1, 4});
    }
}