//
// Created by Amy Fetzner on 11/28/2023.
//

#include "Catch2/catch_test_macros.hpp"
#include "TensorII/Tensor.h"

using namespace TensorII::Core;

SCENARIO("A Tensor can be initialized using an array", "[Tensor]"){
    WHEN("A 0-tensor is initialized by array initialization") {
        Tensor tensor = Tensor<int, Shape{}> {42};
        THEN("It has rank 0") {
            CHECK(tensor.rank() == 0);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 1);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == sizeof(int));
        }
        THEN("It contains the number") {
            CHECK(tensor.data()[0] == 42);
        }
    }

    WHEN("A 1-tensor is initialized by passing an array for initialization") {
        Tensor tensor = Tensor<int, Shape{4}> ({10, 20, 30, 40});
        THEN("It has rank 1") {
            CHECK(tensor.rank() == 1);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{4});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 4);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 4 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0] == 10);
            CHECK(tensor.data()[1] == 20);
            CHECK(tensor.data()[2] == 30);
            CHECK(tensor.data()[3] == 40);
        }
    }

    WHEN("A 2-tensor is initialized by passing an array for initialization") {
        Tensor tensor = Tensor<int, Shape{4, 3}> ({{00, 01, 02},
                                                   {10, 11, 12},
                                                   {20, 21, 22},
                                                   {30, 31, 32}});
        THEN("It has rank 2") {
            CHECK(tensor.rank() == 2);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{4, 3});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 4 * 3);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 4 * 3 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]  == 00);
            CHECK(tensor.data()[1]  == 01);
            CHECK(tensor.data()[2]  == 02);
            CHECK(tensor.data()[3]  == 10);
            CHECK(tensor.data()[4]  == 11);
            CHECK(tensor.data()[5]  == 12);
            CHECK(tensor.data()[6]  == 20);
            CHECK(tensor.data()[7]  == 21);
            CHECK(tensor.data()[8]  == 22);
            CHECK(tensor.data()[9]  == 30);
            CHECK(tensor.data()[10] == 31);
            CHECK(tensor.data()[11] == 32);
        }
    }

    WHEN("A 2-tensor is initialized by passing an l-value array for initialization") {
        int arr[4][3] = {{00, 01, 02},
                         {10, 11, 12},
                         {20, 21, 22},
                         {30, 31, 32}};
        auto tensor = Tensor<int, Shape{4, 3}> (arr);
        THEN("It has rank 2") {
            CHECK(tensor.rank() == 2);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{4, 3});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 4 * 3);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 4 * 3 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]  == 00);
            CHECK(tensor.data()[1]  == 01);
            CHECK(tensor.data()[2]  == 02);
            CHECK(tensor.data()[3]  == 10);
            CHECK(tensor.data()[4]  == 11);
            CHECK(tensor.data()[5]  == 12);
            CHECK(tensor.data()[6]  == 20);
            CHECK(tensor.data()[7]  == 21);
            CHECK(tensor.data()[8]  == 22);
            CHECK(tensor.data()[9]  == 30);
            CHECK(tensor.data()[10] == 31);
            CHECK(tensor.data()[11] == 32);
        }
    }

    WHEN("A 3-tensor is initialized by passing an array for initialization") {
        Tensor tensor = Tensor<int, Shape{3, 2, 1}> ({{{000}, {010}},
                                                      {{100}, {110}},
                                                      {{200}, {210}}});
        THEN("It has rank 3") {
            CHECK(tensor.rank() == 3);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{3, 2, 1});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 3 * 2 * 1);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 3 * 2 * 1 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]  == 000);
            CHECK(tensor.data()[1]  == 010);
            CHECK(tensor.data()[2]  == 100);
            CHECK(tensor.data()[3]  == 110);
            CHECK(tensor.data()[4]  == 200);
            CHECK(tensor.data()[5]  == 210);
        }
    }

    WHEN("A 4-tensor is initialized by passing an array for initialization") {
        Tensor tensor = Tensor<int, Shape{2, 3, 2, 1}> ({{{{0000}, {0010}},
                                                           {{0100}, {0110}},
                                                           {{0200}, {0210}}},
                                                          {{{1000}, {1010}},
                                                           {{1100}, {1110}},
                                                           {{1200}, {1210}}}});
        THEN("It has rank 4") {
            CHECK(tensor.rank() == 4);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{2, 3, 2, 1});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 2 * 3 * 2 * 1);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 2 * 3 * 2 * 1 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]   == 0000);
            CHECK(tensor.data()[1]   == 0010);
            CHECK(tensor.data()[2]   == 0100);
            CHECK(tensor.data()[3]   == 0110);
            CHECK(tensor.data()[4]   == 0200);
            CHECK(tensor.data()[5]   == 0210);
            CHECK(tensor.data()[6]   == 1000);
            CHECK(tensor.data()[7]   == 1010);
            CHECK(tensor.data()[8]   == 1100);
            CHECK(tensor.data()[9]   == 1110);
            CHECK(tensor.data()[10]  == 1200);
            CHECK(tensor.data()[11]  == 1210);
        }
    }
}

/*
SCENARIO("A Tensor can be initialized using an array in a constant expression", "[Tensor][static]"){
    WHEN("A 0-tensor is initialized by array initialization") {
        constexpr Tensor tensor = Tensor<int, Shape{}> {42};
        THEN("It has rank 0") {
            STATIC_CHECK(tensor.rank() == 0);
        }
        THEN("It has the correct shape") {
            STATIC_CHECK(tensor.shape() == Shape{});
        }
        THEN("It has the correct size") {
            STATIC_CHECK(tensor.size() == 1);
        }
        THEN("It has the correct size in bytes") {
            STATIC_CHECK(tensor.size_in_bytes() == sizeof(int));
        }
        THEN("It contains the number") {
            STATIC_CHECK(tensor.data()[0] == 42);
        }
    }

    WHEN("A 1-tensor is initialized by passing an array for initialization") {
        constexpr Tensor tensor = Tensor<int, Shape{4}> ({10, 20, 30, 40});
        THEN("It has rank 1") {
            STATIC_CHECK(tensor.rank() == 1);
        }
        THEN("It has the correct shape") {
            STATIC_CHECK(tensor.shape() == Shape{4});
        }
        THEN("It has the correct size") {
            STATIC_CHECK(tensor.size() == 4);
        }
        THEN("It has the correct size in bytes") {
            STATIC_CHECK(tensor.size_in_bytes() == 4 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            STATIC_CHECK(tensor.data()[0] == 10);
            STATIC_CHECK(tensor.data()[1] == 20);
            STATIC_CHECK(tensor.data()[2] == 30);
            STATIC_CHECK(tensor.data()[3] == 40);
        }
    }

    WHEN("A 2-tensor is initialized by passing an array for initialization") {
        constexpr Tensor tensor = Tensor<int, Shape{4, 3}> ({{00, 01, 02},
                                                             {10, 11, 12},
                                                             {20, 21, 22},
                                                             {30, 31, 32}});
        THEN("It has rank 2") {
            STATIC_CHECK(tensor.rank() == 2);
        }
        THEN("It has the correct shape") {
            STATIC_CHECK(tensor.shape() == Shape{4, 3});
        }
        THEN("It has the correct size") {
            STATIC_CHECK(tensor.size() == 4 * 3);
        }
        THEN("It has the correct size in bytes") {
            STATIC_CHECK(tensor.size_in_bytes() == 4 * 3 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
//            STATIC_CHECK(tensor.data()[0]  == 00);
//            STATIC_CHECK(tensor.data()[1]  == 01);
//            STATIC_CHECK(tensor.data()[2]  == 02);
//            STATIC_CHECK(tensor.data()[3]  == 10);
//            STATIC_CHECK(tensor.data()[4]  == 11);
//            STATIC_CHECK(tensor.data()[5]  == 12);
//            STATIC_CHECK(tensor.data()[6]  == 20);
//            STATIC_CHECK(tensor.data()[7]  == 21);
//            STATIC_CHECK(tensor.data()[8]  == 22);
//            STATIC_CHECK(tensor.data()[9]  == 30);
//            STATIC_CHECK(tensor.data()[10] == 31);
//            STATIC_CHECK(tensor.data()[11] == 32);
        }
    }

    WHEN("A 2-tensor is initialized by passing an l-value array for initialization") {
        constexpr int arr[4][3] = {{00, 01, 02},
                                   {10, 11, 12},
                                   {20, 21, 22},
                                   {30, 31, 32}};
        constexpr auto tensor = Tensor<int, Shape{4, 3}> (arr);
        THEN("It has rank 2") {
            STATIC_CHECK(tensor.rank() == 2);
        }
        THEN("It has the correct shape") {
            STATIC_CHECK(tensor.shape() == Shape{4, 3});
        }
        THEN("It has the correct size") {
            STATIC_CHECK(tensor.size() == 4 * 3);
        }
        THEN("It has the correct size in bytes") {
            STATIC_CHECK(tensor.size_in_bytes() == 4 * 3 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            STATIC_CHECK(tensor.data()[0]  == 00);
            STATIC_CHECK(tensor.data()[1]  == 01);
            STATIC_CHECK(tensor.data()[2]  == 02);
            STATIC_CHECK(tensor.data()[3]  == 10);
            STATIC_CHECK(tensor.data()[4]  == 11);
            STATIC_CHECK(tensor.data()[5]  == 12);
            STATIC_CHECK(tensor.data()[6]  == 20);
            STATIC_CHECK(tensor.data()[7]  == 21);
            STATIC_CHECK(tensor.data()[8]  == 22);
            STATIC_CHECK(tensor.data()[9]  == 30);
            STATIC_CHECK(tensor.data()[10] == 31);
            STATIC_CHECK(tensor.data()[11] == 32);
        }
    }

    WHEN("A 3-tensor is initialized by passing an array for initialization") {
        constexpr Tensor tensor = Tensor<int, Shape{3, 2, 1}> ({{{000}, {010}},
                                                                {{100}, {110}},
                                                                {{200}, {210}}});
        THEN("It has rank 3") {
            STATIC_CHECK(tensor.rank() == 3);
        }
        THEN("It has the correct shape") {
            STATIC_CHECK(tensor.shape() == Shape{3, 2, 1});
        }
        THEN("It has the correct size") {
            STATIC_CHECK(tensor.size() == 3 * 2 * 1);
        }
        THEN("It has the correct size in bytes") {
            STATIC_CHECK(tensor.size_in_bytes() == 3 * 2 * 1 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
//            STATIC_CHECK(tensor.data()[0]  == 000);
//            STATIC_CHECK(tensor.data()[1]  == 010);
//            STATIC_CHECK(tensor.data()[2]  == 100);
//            STATIC_CHECK(tensor.data()[3]  == 110);
//            STATIC_CHECK(tensor.data()[4]  == 200);
//            STATIC_CHECK(tensor.data()[5]  == 210);
        }
    }

    WHEN("A 4-tensor is initialized by passing an array for initialization") {
        constexpr Tensor tensor = Tensor<int, Shape{2, 3, 2, 1}> ({{{{0000}, {0010}},
                                                                    {{0100}, {0110}},
                                                                    {{0200}, {0210}}},
                                                                   {{{1000}, {1010}},
                                                                    {{1100}, {1110}},
                                                                    {{1200}, {1210}}}});
        THEN("It has rank 4") {
            STATIC_CHECK(tensor.rank() == 4);
        }
        THEN("It has the correct shape") {
            STATIC_CHECK(tensor.shape() == Shape{2, 3, 2, 1});
        }
        THEN("It has the correct size") {
            STATIC_CHECK(tensor.size() == 2 * 3 * 2 * 1);
        }
        THEN("It has the correct size in bytes") {
            STATIC_CHECK(tensor.size_in_bytes() == 2 * 3 * 2 * 1 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
//            STATIC_CHECK(tensor.data()[0]   == 0000);
//            STATIC_CHECK(tensor.data()[1]   == 0010);
//            STATIC_CHECK(tensor.data()[2]   == 0100);
//            STATIC_CHECK(tensor.data()[3]   == 0110);
//            STATIC_CHECK(tensor.data()[4]   == 0200);
//            STATIC_CHECK(tensor.data()[5]   == 0210);
//            STATIC_CHECK(tensor.data()[6]   == 1000);
//            STATIC_CHECK(tensor.data()[7]   == 1010);
//            STATIC_CHECK(tensor.data()[8]   == 1100);
//            STATIC_CHECK(tensor.data()[9]   == 1110);
//            STATIC_CHECK(tensor.data()[10]  == 1200);
//            STATIC_CHECK(tensor.data()[11]  == 1210);
        }
    }
}
*/

SCENARIO("A Tensor can be initialized using toTensor", "[Tensor]"){
    WHEN("A 0-tensor is initialized with toTensor") {
        Tensor tensor = toTensor<int>(42);
        THEN("It contains the number") {
            CHECK(tensor.data()[0] == 42);
        }
        THEN("It has rank 0") {
            CHECK(tensor.rank() == 0);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 1);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == sizeof(int));
        }
        THEN("It contains the number") {
            CHECK(tensor.data()[0] == 42);
        }
    }

    WHEN("A 1-tensor is initialized with toTensor") {
        Tensor tensor = toTensor<int> ({10, 20, 30, 40});
        THEN("It has rank 1") {
            CHECK(tensor.rank() == 1);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{4});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 4);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 4 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0] == 10);
            CHECK(tensor.data()[1] == 20);
            CHECK(tensor.data()[2] == 30);
            CHECK(tensor.data()[3] == 40);
        }
    }

    WHEN("A 2-tensor is initialized with toTensor and an l-value array") {
        int arr[4][3] = {{00, 01, 02},
                         {10, 11, 12},
                         {20, 21, 22},
                         {30, 31, 32}};
        Tensor tensor = toTensor<int> (arr);
        THEN("It has rank 4") {
            CHECK(tensor.rank() == 2);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{4, 3});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 4 * 3);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 4 * 3 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]  == 00);
            CHECK(tensor.data()[1]  == 01);
            CHECK(tensor.data()[2]  == 02);
            CHECK(tensor.data()[3]  == 10);
            CHECK(tensor.data()[4]  == 11);
            CHECK(tensor.data()[5]  == 12);
            CHECK(tensor.data()[6]  == 20);
            CHECK(tensor.data()[7]  == 21);
            CHECK(tensor.data()[8]  == 22);
            CHECK(tensor.data()[9]  == 30);
            CHECK(tensor.data()[10] == 31);
            CHECK(tensor.data()[11] == 32);
        }
    }

    WHEN("A 2-tensor is initialized with toTensor") {
        Tensor tensor = toTensor<int> ({{00, 01, 02},
                                        {10, 11, 12},
                                        {20, 21, 22},
                                        {30, 31, 32}});
        THEN("It has rank 4") {
            CHECK(tensor.rank() == 2);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{4, 3});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 4 * 3);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 4 * 3 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]  == 00);
            CHECK(tensor.data()[1]  == 01);
            CHECK(tensor.data()[2]  == 02);
            CHECK(tensor.data()[3]  == 10);
            CHECK(tensor.data()[4]  == 11);
            CHECK(tensor.data()[5]  == 12);
            CHECK(tensor.data()[6]  == 20);
            CHECK(tensor.data()[7]  == 21);
            CHECK(tensor.data()[8]  == 22);
            CHECK(tensor.data()[9]  == 30);
            CHECK(tensor.data()[10] == 31);
            CHECK(tensor.data()[11] == 32);
        }
    }

    WHEN("A 3-tensor is initialized with toTensor") {
        Tensor tensor = toTensor<int> ({{{000}, {010}},
                                        {{100}, {110}},
                                        {{200}, {210}}});
        THEN("It has rank 3") {
            CHECK(tensor.rank() == 3);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{3, 2, 1});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 3 * 2 * 1);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 3 * 2 * 1 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]  == 000);
            CHECK(tensor.data()[1]  == 010);
            CHECK(tensor.data()[2]  == 100);
            CHECK(tensor.data()[3]  == 110);
            CHECK(tensor.data()[4]  == 200);
            CHECK(tensor.data()[5]  == 210);
        }
    }

    WHEN("A 4-tensor is initialized with toTensor") {
        Tensor tensor = toTensor<int> ({{{{0000}, {0010}},
                                         {{0100}, {0110}},
                                         {{0200}, {0210}}},
                                        {{{1000}, {1010}},
                                         {{1100}, {1110}},
                                         {{1200}, {1210}}}});
        THEN("It has rank 4") {
            CHECK(tensor.rank() == 4);
        }
        THEN("It has the correct shape") {
            CHECK(tensor.shape() == Shape{2, 3, 2, 1});
        }
        THEN("It has the correct size") {
            CHECK(tensor.size() == 2 * 3 * 2 * 1);
        }
        THEN("It has the correct size in bytes") {
            CHECK(tensor.size_in_bytes() == 2 * 3 * 2 * 1 * sizeof(int));
        }
        THEN("It contains the correct numbers") {
            CHECK(tensor.data()[0]   == 0000);
            CHECK(tensor.data()[1]   == 0010);
            CHECK(tensor.data()[2]   == 0100);
            CHECK(tensor.data()[3]   == 0110);
            CHECK(tensor.data()[4]   == 0200);
            CHECK(tensor.data()[5]   == 0210);
            CHECK(tensor.data()[6]   == 1000);
            CHECK(tensor.data()[7]   == 1010);
            CHECK(tensor.data()[8]   == 1100);
            CHECK(tensor.data()[9]   == 1110);
            CHECK(tensor.data()[10]  == 1200);
            CHECK(tensor.data()[11]  == 1210);
        }
    }
}

// TODO: Check constant expression toTensor

struct CopyCounter {
    int value;
    int count;

    CopyCounter(int value = 0, int count = 0) // NOLINT(google-explicit-constructor)
        : value(value), count(count) {}

    CopyCounter(const CopyCounter& other)
        : value(other.value), count(other.count + 1)
    {}

    CopyCounter& operator=(const CopyCounter& other){
        value = other.value;
        count = other.count + 1;
        return *this;
    }
    // Required so that CopyCounter satisfies Scalar requirement
    friend CopyCounter operator+(const CopyCounter& a, const CopyCounter& b){
        return {a.value + b.value, -1};
    }
    friend CopyCounter operator-(const CopyCounter& a, const CopyCounter& b){
        return {a.value - b.value, -1};
    }
    friend CopyCounter operator*(const CopyCounter& a, const CopyCounter& b){
        return {a.value * b.value, -1};
    }
    friend CopyCounter operator/(const CopyCounter& a, const CopyCounter& b){
        return {a.value / b.value, -1};
    }
};

SCENARIO("Initializing a tensor minimally copies the array", "[!mayfail]") {
    CopyCounter k = 42;
    WHEN ("CopyCounter is copy constructed"){
        auto copied = CopyCounter(k);
        THEN("Its value stays the same, and its count increments"){
            REQUIRE(k.value == 42);
            REQUIRE(k.count == 0);
            REQUIRE(copied.value == 42);
            REQUIRE(copied.count == 1);
        }
    }
    WHEN ("CopyCounter is copy assigned"){
        CopyCounter copied = k;
        THEN("Its value stays the same, and its count increments"){
            REQUIRE(k.value == 42);
            REQUIRE(k.count == 0);
            REQUIRE(copied.value == 42);
            REQUIRE(copied.count == 1);
        }
    }

    WHEN ("A tensor is created from an r-value array") {
        Tensor tensor = Tensor<CopyCounter, Shape{4, 3}> ({{CopyCounter{00}, CopyCounter{01}, CopyCounter{02}},
                                                           {CopyCounter{10}, CopyCounter{11}, CopyCounter{12}},
                                                           {CopyCounter{20}, CopyCounter{21}, CopyCounter{22}},
                                                           {CopyCounter{30}, CopyCounter{31}, CopyCounter{32}}});
        THEN("Each element was mem-moved without copying") {
            CHECK(tensor.data()[0].value == 00);
            CHECK(tensor.data()[0].count == 1);
            CHECK(tensor.data()[3].value == 10);
            CHECK(tensor.data()[3].count == 1);
            CHECK(tensor.data()[7].value == 21);
            CHECK(tensor.data()[7].count == 1);
        }
    }

    WHEN ("A tensor is created from an l-value array") {
        CopyCounter arr[4][3] = {{CopyCounter{00}, CopyCounter{01}, CopyCounter{02}},
                                 {CopyCounter{10}, CopyCounter{11}, CopyCounter{12}},
                                 {CopyCounter{20}, CopyCounter{21}, CopyCounter{22}},
                                 {CopyCounter{30}, CopyCounter{31}, CopyCounter{32}}};
        auto tensor = Tensor<CopyCounter, Shape{4, 3}> (arr);
        THEN("Each element was mem-moved without copying") {
            CHECK(tensor.data()[0].value == 00);
            CHECK(tensor.data()[0].count == 1);
            CHECK(tensor.data()[3].value == 10);
            CHECK(tensor.data()[3].count == 1);
            CHECK(tensor.data()[7].value == 21);
            CHECK(tensor.data()[7].count == 1);
        }
    }

    WHEN ("A tensor is created using toTensor and an r-value array") {
        Tensor tensor = toTensor<CopyCounter> ({{00, 01, 02},
                                                {10, 11, 12},
                                                {20, 21, 22},
                                                {30, 31, 32}});
        THEN("Each element was only copied once") {
            CHECK(tensor.data()[0].value == 00);
            CHECK(tensor.data()[0].count == 1);
            CHECK(tensor.data()[3].value == 10);
            CHECK(tensor.data()[3].count == 1);
            CHECK(tensor.data()[7].value == 21);
            CHECK(tensor.data()[7].count == 1);
        }
    }

    WHEN ("A tensor is created using toTensor and an l-value array") {
        CopyCounter arr[4][3] = {{00, 01, 02},
                                 {10, 11, 12},
                                 {20, 21, 22},
                                 {30, 31, 32}};
        Tensor tensor = toTensor<CopyCounter> (arr);
        THEN("Each element was only copied once") {
            CHECK(tensor.data()[0].value == 00);
            CHECK(tensor.data()[0].count == 1);
            CHECK(tensor.data()[3].value == 10);
            CHECK(tensor.data()[3].count == 1);
            CHECK(tensor.data()[7].value == 21);
            CHECK(tensor.data()[7].count == 1);
        }
    }
}

TEST_CASE("Tensor Reshape", "[Tensor]") {
    using namespace TensorII::Core;
    Tensor<int, Shape{2, 3}> t23 ({{1, 2, 3}, {4, 5, 6}});
//    auto t32 = reshape<Shape<3, 2>>(t23);
}