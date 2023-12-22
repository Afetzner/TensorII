//
// Created by Amy Fetzner on 12/21/2023.
//

#include "TensorII/TensorInitializer.h"
#include "Catch2/catch_test_macros.hpp"

using namespace TensorII::Core;
using namespace TensorII::Core::Private;

TEST_CASE("Tensor Initializer, 0D", "[TensorInit]"){
    constexpr TensorInitializer<int, Shape{}> tensorInit {1};
    CHECK(*tensorInit.begin() == 1);
    CHECK(++tensorInit.begin() == tensorInit.end());
}

TEST_CASE("Tensor Initializer, 1D - dynamic", "[TensorInit]"){
    int arr[] = {1, 2, 3, 4};
    TensorInitializer<int, Shape{4}> tensorInit {arr};
    CHECK(std::ranges::equal(arr, tensorInit));
}

TEST_CASE("Tensor Initializer, 1D - static", "[TensorInit]"){
    static constexpr int arr[] = {1, 2, 3, 4};
    constexpr TensorInitializer<int, Shape{4}> tensorInit {arr};
    STATIC_CHECK(std::ranges::equal(arr, tensorInit));
}

TEST_CASE("Tensor Initializer, 2D - dynamic", "[TensorInit]"){
    int arr[2][4] = {{1, 2, 3, 4},
                     {5, 6, 7, 8}};
    int flat[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    TensorInitializer<int, Shape{2, 4}> tensorInit {arr};
    CHECK(std::ranges::equal(tensorInit, flat));
}

TEST_CASE("Tensor Initializer, 2D - static", "[TensorInit]"){
    static constexpr int arr[2][4] = {{1, 2, 3, 4},
                                      {5, 6, 7, 8}};
    static constexpr int flat[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    constexpr TensorInitializer<int, Shape{2, 4}> tensorInit {arr};
    STATIC_CHECK(std::ranges::equal(tensorInit, flat));
}

TEST_CASE("Tensor Initializer, 3D - dynamic", "[TensorInit]"){
    int arr[2][4][3] = {{{1, 10, 100}, {2, 20, 200}, {3, 30, 300}, {4, 40, 400}},
                        {{5, 50, 500}, {6, 60, 600}, {7, 70, 700}, {8, 80, 800}}};
    int flat[24] = {1, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400,
                    5, 50, 500, 6, 60, 600, 7, 70, 700, 8, 80, 800};
    TensorInitializer<int, Shape{2, 4, 3}> tensorInit {arr};
    CHECK(std::ranges::equal(tensorInit, flat));
}

TEST_CASE("Tensor Initializer, 3D - static", "[TensorInit]"){
    static constexpr int arr[2][4][3] = {{{1, 10, 100}, {2, 20, 200}, {3, 30, 300}, {4, 40, 400}},
                                         {{5, 50, 500}, {6, 60, 600}, {7, 70, 700}, {8, 80, 800}}};
    static constexpr int flat[24] = {1, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400,
                                     5, 50, 500, 6, 60, 600, 7, 70, 700, 8, 80, 800};
    constexpr TensorInitializer<int, Shape{2, 4, 3}> tensorInit {arr};
    STATIC_CHECK(std::ranges::equal(tensorInit, flat));
}
