//
// Created by Amy Fetzner on 11/28/2023.
//

#include "Catch2/catch_test_macros.hpp"
#include "TensorII/Tensor.h"

using namespace TensorII::Core;

TEST_CASE("Tensor Initialization", "[Tensor][Init]"){
    Tensor t0 = Tensor<int, Shape<>> (1); // NOLINT(modernize-use-auto)
    CHECK(*t0.data() == 1);

    const int expected[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor t1 = Tensor<int, Shape<3>> ({1, 2, 3});
    REQUIRE(t1.size() == 3);
    CHECK(memcmp(t1.data(), expected, 3) == 0);

    Tensor t2 = Tensor<int, Shape<2, 3>> ({{1, 2, 3},
                                           {4, 5, 6}});
    REQUIRE(t2.size() == 6);
    CHECK(memcmp(t2.data(), expected, 6) == 0);

    Tensor t3 = Tensor<int, Shape<2, 2, 3>> ({
                                             {{1 , 2 , 3 }, {4 , 5 , 6 }},
                                             {{7 , 8 , 9 }, {10, 11, 12}}
                                             });
    REQUIRE(t3.size() == 12);
    CHECK(memcmp(t3.data(), expected, 6) == 0);

    Tensor t4 = Tensor<int, Shape<2, 3, 2>> ({
                                             {{1 , 2} , {3 , 4 }, {5 , 6 }},
                                             {{7 , 8} , {9 , 10}, {11, 12}}
                                             });
    REQUIRE(t4.size() == 12);
    CHECK(memcmp(t4.data(), expected, 6) == 0);

    Tensor t5 = Tensor<int, Shape<1, 1, 1, 5>> ({{{{1, 2, 3, 4, 5}}}});
    REQUIRE(t5.size() == 5);
    CHECK(memcmp(t5.data(), expected, 5) == 0);
}

TEST_CASE("Tensor ToTensor", "[Tensor][Init]"){
    Tensor u0 = Tensor<int, Shape<>> (1); // NOLINT(modernize-use-auto)
    Tensor t0 = toTensor<int>(1);
    STATIC_CHECK(std::is_same_v<decltype(u0), decltype(t0)>);

    Tensor u1 = Tensor<int, Shape<3>> ({1, 2, 3});
    Tensor t1 = toTensor<int>({1, 2, 3});
    STATIC_CHECK(std::is_same_v<decltype(u1), decltype(t1)>);

    Tensor u2 = Tensor<int, Shape<2, 3>> ({{1, 2, 3},
                                           {4, 5, 6}});
    Tensor t2 = toTensor<int>({{1, 2, 3},
                               {4, 5, 6}});

    STATIC_CHECK(std::is_same_v<decltype(u2), decltype(t2)>);

    Tensor u3 = Tensor<int, Shape<2, 2, 3>> ({
                                             {{1 , 2 , 3 }, {4 , 5 , 6 }},
                                             {{7 , 8 , 9 }, {10, 11, 12}}
                                             });
    Tensor t3 = toTensor<int>({
                              {{1 , 2 , 3 }, {4 , 5 , 6 }},
                              {{7 , 8 , 9 }, {10, 11, 12}}
                              });

    Tensor u4 = Tensor<int, Shape<2, 3, 2>> ({
                                             {{1 , 2} , {3 , 4 }, {5 , 6 }},
                                             {{7 , 8} , {9 , 10}, {11, 12}}
                                             });
    Tensor t4 = toTensor<int>({
                              {{1 , 2} , {3 , 4 }, {5 , 6 }},
                              {{7 , 8} , {9 , 10}, {11, 12}}
                              });

    Tensor u5 = Tensor<int, Shape<1, 1, 1, 5>> ({{{{1, 2, 3, 4, 5}}}});
    Tensor t5 = toTensor<int> ({{{{1, 2, 3, 4, 5}}}});
    STATIC_CHECK(std::is_same_v<decltype(u5), decltype(t5)>);
}

TEST_CASE("Tensor Reshape", "[Tensor]") {
    using namespace TensorII::Core;
    Tensor<int, Shape<2, 3>> t23 ({{1, 2, 3}, {4, 5, 6}});
//    auto t32 = reshape<Shape<3, 2>>(t23);
}