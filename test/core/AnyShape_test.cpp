//
// Created by Amy Fetzner on 12/10/2023.
//

// NOTE: All equality comparisons inside Catch2 checks in this file must be wrapped in parentheses
// STATIC_CHECK(anyShape0 == Shape{});  -->  STATIC_CHECK((anyShape0 == Shape{}));
// because shapes are ranges, Catch2 is trying to use its own range comparator instead of the defined operator==

#include "TensorII/AnyShape.h"
#include "Catch2/catch_test_macros.hpp"

using namespace TensorII::Core;

TEST_CASE("AnyShape constructor from args", "[AnyShape]"){
    constexpr AnyShape<4> anyShape0{};
    STATIC_CHECK(anyShape0 == Shape{});

    constexpr AnyShape<4> anyShape1{1};
    STATIC_CHECK(anyShape1 == Shape{1});

    constexpr AnyShape<4> anyShape2{1, 2};
    STATIC_CHECK(anyShape2 == Shape{1, 2});

    constexpr AnyShape<4> anyShape3{1, 2, 3};
    STATIC_CHECK(anyShape3 == Shape{1, 2, 3});

    constexpr AnyShape<4> anyShape4{1, 2, 3, 4};
    STATIC_CHECK(anyShape4 == Shape{1, 2, 3, 4});
}

TEST_CASE("AnyShape constructor from shape", "[AnyShape]"){
    constexpr AnyShape<4> anyShape0 = Shape{};
    STATIC_CHECK((anyShape0 == Shape{}));

    constexpr AnyShape<4> anyShape1 = Shape{1};
    STATIC_CHECK((anyShape1 == Shape{1}));

    constexpr AnyShape<4> anyShape2 = Shape{1, 2};
    STATIC_CHECK((anyShape2 == Shape{1, 2}));

    constexpr AnyShape<4> anyShape3 = Shape{1, 2, 3};
    STATIC_CHECK((anyShape3 == Shape{1, 2, 3}));

    constexpr AnyShape<4> anyShape4 = Shape{1, 2, 3, 4};
    STATIC_CHECK((anyShape4 == Shape{1, 2, 3, 4}));
}

TEST_CASE("AnyShape constructor from range, array", "[AnyShape]"){
    std::array<tensorDimension, 1> array1{1};
    AnyShape<4> anyShape1{from_range, array1};
    CHECK((anyShape1 == Shape{1}));

    std::array<tensorDimension, 2> array2{1, 2};
    AnyShape<4> anyShape2{from_range, array2};
    CHECK((anyShape2 == Shape{1, 2}));

    std::array<tensorDimension, 3> array3{1, 2, 3};
    AnyShape<4> anyShape3{from_range, array3};
    CHECK((anyShape3 == Shape{1, 2, 3}));

    std::array<tensorDimension, 4> array4{1, 2, 3, 4};
    AnyShape<4> anyShape4{from_range, array4};
    CHECK((anyShape4 == Shape{1, 2, 3, 4}));
}

TEST_CASE("AnyShape constructor from range, vector", "[AnyShape]"){
    std::vector<tensorDimension> vector0{};
    AnyShape<4> anyShape0{from_range, vector0};
    CHECK((anyShape0 == Shape{}));

    std::vector<tensorDimension> vector1{1};
    AnyShape<4> anyShape1{from_range, vector1};
    CHECK((anyShape1 == Shape{1}));

    std::vector<tensorDimension> vector2{1, 2};
    AnyShape<4> anyShape2{from_range, vector2};
    CHECK((anyShape2 == Shape{1, 2}));

    std::vector<tensorDimension> vector3{1, 2, 3};
    AnyShape<4> anyShape3{from_range, vector3};
    CHECK((anyShape3 == Shape{1, 2, 3}));

    std::vector<tensorDimension> vector4{1, 2, 3, 4};
    AnyShape<4> anyShape4{from_range, vector4};
    CHECK((anyShape4 == Shape{1, 2, 3, 4}));
}

TEST_CASE("AnyShape emplace from args", "[AnyShape]"){
    AnyShape<4> anyShape0;
    anyShape0.emplace();
    CHECK((anyShape0 == Shape{}));

    AnyShape<4> anyShape1;
    anyShape1.emplace(1);
    CHECK((anyShape1 == Shape{1}));

    AnyShape<4> anyShape2;
    anyShape2.emplace(1, 2);
    CHECK((anyShape2 == Shape{1, 2}));

    AnyShape<4> anyShape3;
    anyShape3.emplace(1, 2, 3);
    CHECK((anyShape3 == Shape{1, 2, 3}));

    AnyShape<4> anyShape4;
    anyShape4.emplace(1, 2, 3, 4);
    CHECK((anyShape4 == Shape{1, 2, 3, 4}));

    // Should not compile
//    AnyShape<4> anyShape5;
//    anyShape5.emplace(1, 2, 3, 4, 5);
}

TEST_CASE("AnyShape emplace from range, array", "[AnyShape]"){
    std::array<tensorDimension, 1> array1{1};
    AnyShape<4> anyShape1;
    anyShape1.emplace(from_range, array1);
    CHECK((anyShape1 == Shape{1}));

    std::array<tensorDimension, 2> array2{1, 2};
    AnyShape<4> anyShape2;
    anyShape2.emplace(from_range, array2);
    CHECK((anyShape2 == Shape{1, 2}));

    std::array<tensorDimension, 3> array3{1, 2, 3};
    AnyShape<4> anyShape3;
    anyShape3.emplace(from_range, array3);
    CHECK((anyShape3 == Shape{1, 2, 3}));

    std::array<tensorDimension, 4> array4{1, 2, 3, 4};
    AnyShape<4> anyShape4;
    anyShape4.emplace(from_range, array4);
    CHECK((anyShape4 == Shape{1, 2, 3, 4}));

    std::array<tensorDimension, 5> array5{1, 2, 3, 4, 5};
    AnyShape<4> anyShape5;
    CHECK_THROWS((anyShape5.emplace(from_range, array5)));
}

TEST_CASE("AnyShape emplace from range, vector", "[AnyShape]"){
    std::vector<tensorDimension> vector0{};
    AnyShape<4> anyShape0;
    anyShape0.emplace(from_range, vector0);
    CHECK((anyShape0 == Shape{}));


    std::vector<tensorDimension> vector1{1};
    AnyShape<4> anyShape1;
    anyShape1.emplace(from_range, vector1);
    CHECK((anyShape1 == Shape{1}));

    std::vector<tensorDimension> vector2{1, 2};
    AnyShape<4> anyShape2;
    anyShape2.emplace(from_range, vector2);
    CHECK((anyShape2 == Shape{1, 2}));

    std::vector<tensorDimension> vector3{1, 2, 3};
    AnyShape<4> anyShape3;
    anyShape3.emplace(from_range, vector3);
    CHECK((anyShape3 == Shape{1, 2, 3}));

    std::vector<tensorDimension> vector4{1, 2, 3, 4};
    AnyShape<4> anyShape4;
    anyShape4.emplace(from_range, vector4);
    CHECK((anyShape4 == Shape{1, 2, 3, 4}));

    std::vector<tensorDimension> vector5{1, 2, 3, 4, 5};
    AnyShape<4> anyShape5;
    CHECK_THROWS(anyShape5.emplace(from_range, vector5));
}

TEST_CASE("AnyShape augmented from args", "[AnyShape]") {
    constexpr AnyShape<4> anyShape0{};
    STATIC_CHECK((anyShape0.augmented() == Shape{}));
    STATIC_CHECK((anyShape0.augmented(1) == Shape{1}));
    STATIC_CHECK((anyShape0.augmented(1, 2) == Shape{1, 2}));
    STATIC_CHECK((anyShape0.augmented(1, 2, 3) == Shape{1, 2, 3}));
    STATIC_CHECK((anyShape0.augmented(1, 2, 3, 4) == Shape{1, 2, 3, 4}));
    CHECK_THROWS(anyShape0.augmented(1, 2, 3, 4, 5));

    constexpr AnyShape<4> anyShape1{10};
    STATIC_CHECK((anyShape1.augmented() == Shape{10}));
    STATIC_CHECK((anyShape1.augmented(1) == Shape{10, 1}));
    STATIC_CHECK((anyShape1.augmented(1, 2) == Shape{10, 1, 2}));
    STATIC_CHECK((anyShape1.augmented(1, 2, 3) == Shape{10, 1, 2, 3}));
    CHECK_THROWS(anyShape1.augmented(1, 2, 3, 4));

    constexpr AnyShape<4> anyShape2{10, 20};
    STATIC_CHECK((anyShape2.augmented() == Shape{10, 20}));
    STATIC_CHECK((anyShape2.augmented(1) == Shape{10, 20, 1}));
    STATIC_CHECK((anyShape2.augmented(1, 2) == Shape{10, 20, 1, 2}));
    CHECK_THROWS(anyShape2.augmented(1, 2, 3));

    constexpr AnyShape<4> anyShape3{10, 20, 30};
    STATIC_CHECK((anyShape3.augmented() == Shape{10, 20, 30}));
    STATIC_CHECK((anyShape3.augmented(1) == Shape{10, 20, 30, 1}));
    CHECK_THROWS(anyShape3.augmented(1, 2));

    constexpr AnyShape<4> anyShape4{10, 20, 30, 40};
    STATIC_CHECK((anyShape4.augmented() == Shape{10, 20, 30, 40}));
    CHECK_THROWS((anyShape4.augmented(1)));
}

TEST_CASE("AnyShape augmented from range - array", "[AnyShape]") {
    constexpr std::array<tensorDimension, 1> array1{1};
    constexpr std::array<tensorDimension, 2> array2{1, 2};
    constexpr std::array<tensorDimension, 3> array3{1, 2, 3};
    constexpr std::array<tensorDimension, 4> array4{1, 2, 3, 4};
    constexpr std::array<tensorDimension, 5> array5{1, 2, 3, 4, 5};


    constexpr AnyShape<4> anyShape0{};
    STATIC_CHECK(anyShape0.augmented(from_range, array1) == Shape{1});
    STATIC_CHECK(anyShape0.augmented(from_range, array2) == Shape{1, 2});
    STATIC_CHECK(anyShape0.augmented(from_range, array3) == Shape{1, 2, 3});
    STATIC_CHECK(anyShape0.augmented(from_range, array4) == Shape{1, 2, 3, 4});
    CHECK_THROWS(anyShape0.augmented(from_range, array5));

    constexpr AnyShape<4> anyShape1{10};
    STATIC_CHECK(anyShape1.augmented(from_range, array1) == Shape{10, 1});
    STATIC_CHECK(anyShape1.augmented(from_range, array2) == Shape{10, 1, 2});
    STATIC_CHECK(anyShape1.augmented(from_range, array3) == Shape{10, 1, 2, 3});
    CHECK_THROWS(anyShape1.augmented(from_range, array4));

    constexpr AnyShape<4> anyShape2{10, 20};
    STATIC_CHECK(anyShape2.augmented(from_range, array1) == Shape{10, 20, 1});
    STATIC_CHECK(anyShape2.augmented(from_range, array2) == Shape{10, 20, 1, 2});
    CHECK_THROWS(anyShape2.augmented(from_range, array3));

    constexpr AnyShape<4> anyShape3{10, 20, 30};
    STATIC_CHECK(anyShape3.augmented(from_range, array1) == Shape{10, 20, 30, 1});
    CHECK_THROWS(anyShape3.augmented(from_range, array2));

    constexpr AnyShape<4> anyShape4{10, 20, 30, 40};
    CHECK_THROWS(anyShape4.augmented(from_range, array1));
}

TEST_CASE("AnyShape augmented from range - vector", "[AnyShape]") {
    std::vector<tensorDimension> vector0{};
    std::vector<tensorDimension> vector1{1};
    std::vector<tensorDimension> vector2{1, 2};
    std::vector<tensorDimension> vector3{1, 2, 3};
    std::vector<tensorDimension> vector4{1, 2, 3, 4};
    std::vector<tensorDimension> vector5{1, 2, 3, 4, 5};


    AnyShape<4> anyShape0{};
    CHECK((anyShape0.augmented(from_range, vector0) == Shape{}));
    CHECK((anyShape0.augmented(from_range, vector1) == Shape{1}));
    CHECK((anyShape0.augmented(from_range, vector2) == Shape{1, 2}));
    CHECK((anyShape0.augmented(from_range, vector3) == Shape{1, 2, 3}));
    CHECK((anyShape0.augmented(from_range, vector4) == Shape{1, 2, 3, 4}));
    CHECK_THROWS(anyShape0.augmented(from_range, vector5));

    AnyShape<4> anyShape1{10};
    CHECK((anyShape1.augmented(from_range, vector0) == Shape{10}));
    CHECK((anyShape1.augmented(from_range, vector1) == Shape{10, 1}));
    CHECK((anyShape1.augmented(from_range, vector2) == Shape{10, 1, 2}));
    CHECK((anyShape1.augmented(from_range, vector3) == Shape{10, 1, 2, 3}));
    CHECK_THROWS(anyShape1.augmented(from_range, vector4));

    AnyShape<4> anyShape2{10, 20};
    CHECK((anyShape2.augmented(from_range, vector0) == Shape{10, 20}));
    CHECK((anyShape2.augmented(from_range, vector1) == Shape{10, 20, 1}));
    CHECK((anyShape2.augmented(from_range, vector2) == Shape{10, 20, 1, 2}));
    CHECK_THROWS(anyShape2.augmented(from_range, vector3));

    AnyShape<4> anyShape3{10, 20, 30};
    CHECK((anyShape3.augmented(from_range, vector0) == Shape{10, 20, 30}));
    CHECK((anyShape3.augmented(from_range, vector1) == Shape{10, 20, 30, 1}));
    CHECK_THROWS(anyShape3.augmented(from_range, vector2));

    AnyShape<4> anyShape4{10, 20, 30, 40};
    CHECK((anyShape4.augmented(from_range, vector0) == Shape{10, 20, 30, 40}));
    CHECK_THROWS(anyShape4.augmented(from_range, vector1));
}

TEST_CASE("AnyShape demoted", "[AnyShape]") {
    constexpr AnyShape<4> anyShape4{1, 2, 3, 4};
    STATIC_CHECK((anyShape4.demoted(4) == Shape{1, 2, 3, 4}));
    STATIC_CHECK((anyShape4.demoted(3) == Shape{1, 2, 3}));
    STATIC_CHECK((anyShape4.demoted(2) == Shape{1, 2}));
    STATIC_CHECK((anyShape4.demoted(1) == Shape{1}));
    STATIC_CHECK((anyShape4.demoted(0) == Shape{}));

    constexpr AnyShape<4> anyShape3{1, 2, 3};
    CHECK_THROWS(anyShape3.demoted(4));
    STATIC_CHECK((anyShape3.demoted(3) == Shape{1, 2, 3}));
    STATIC_CHECK((anyShape3.demoted(2) == Shape{1, 2}));
    STATIC_CHECK((anyShape3.demoted(1) == Shape{1}));
    STATIC_CHECK((anyShape3.demoted(0) == Shape{}));

    constexpr AnyShape<4> anyShape2{1, 2};
    CHECK_THROWS(anyShape2.demoted(4));
    CHECK_THROWS(anyShape2.demoted(3));
    STATIC_CHECK((anyShape2.demoted(2) == Shape{1, 2}));
    STATIC_CHECK((anyShape2.demoted(1) == Shape{1}));
    STATIC_CHECK((anyShape2.demoted(0) == Shape{}));

    constexpr AnyShape<4> anyShape1{1};
    CHECK_THROWS(anyShape1.demoted(4));
    CHECK_THROWS(anyShape1.demoted(3));
    CHECK_THROWS(anyShape1.demoted(2));
    STATIC_CHECK((anyShape1.demoted(1) == Shape{1}));
    STATIC_CHECK((anyShape1.demoted(0) == Shape{}));

    constexpr AnyShape<4> anyShape0{};
    CHECK_THROWS(anyShape0.demoted(4));
    CHECK_THROWS(anyShape0.demoted(3));
    CHECK_THROWS(anyShape0.demoted(2));
    CHECK_THROWS(anyShape0.demoted(1));
    STATIC_CHECK((anyShape0.demoted(0) == Shape{}));
}

TEST_CASE("AnyShape augment args", "[AnyShape]") {
    AnyShape<10> anyShape0{};
    anyShape0.augment();
    CHECK((anyShape0 == Shape{}));
    anyShape0.augment(1);
    CHECK((anyShape0 == Shape{1}));
    anyShape0.augment(2, 3);
    CHECK((anyShape0 == Shape{1, 2, 3}));
    anyShape0.augment(4, 5, 6);
    CHECK((anyShape0 == Shape{1, 2, 3, 4, 5, 6}));

    AnyShape<10> anyShape4{1, 2, 3, 4};
    anyShape4.augment();
    CHECK((anyShape4 == Shape{1, 2, 3, 4}));
    anyShape4.augment(5, 6, 7, 8, 9, 10);
    CHECK((anyShape4 == Shape{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    anyShape4.augment();
    CHECK((anyShape4 == Shape{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    CHECK_THROWS(anyShape4.augment(1));
}

TEST_CASE("AnyShape augment from range vector", "[AnyShape]") {
    AnyShape<10> anyShape0{};
    anyShape0.augment(from_range, std::vector<tensorDimension>{});
    CHECK((anyShape0 == Shape{}));
    anyShape0.augment(from_range, std::vector<tensorDimension>{1});
    CHECK((anyShape0 == Shape{1}));
    anyShape0.augment(from_range, std::vector<tensorDimension>{2, 3});
    CHECK((anyShape0 == Shape{1, 2, 3}));
    anyShape0.augment(from_range, std::vector<tensorDimension>{4, 5, 6});
    CHECK((anyShape0 == Shape{1, 2, 3, 4, 5, 6}));

    AnyShape<10> anyShape4{1, 2, 3, 4};
    anyShape4.augment(from_range, std::vector<tensorDimension>{});
    CHECK((anyShape4 == Shape{1, 2, 3, 4}));
    anyShape4.augment(from_range, std::vector<tensorDimension>{5, 6, 7, 8, 9, 10});
    CHECK((anyShape4 == Shape{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    anyShape4.augment(from_range, std::vector<tensorDimension>{});
    CHECK((anyShape4 == Shape{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    CHECK_THROWS(anyShape4.augment(from_range, std::vector<tensorDimension>{1}));
}

TEST_CASE("AnyShape demote", "[AnyShape]") {
    {
        AnyShape<4> anyShape4{1, 2, 3, 4};
        anyShape4.demote(4);
        CHECK((anyShape4 == Shape{1, 2, 3, 4}));
        anyShape4.demote(3);
        CHECK((anyShape4 == Shape{1, 2, 3}));
        anyShape4.demote(2);
        CHECK((anyShape4 == Shape{1, 2}));
        anyShape4.demote(1);
        CHECK((anyShape4 == Shape{1}));
        anyShape4.demote(0);
        CHECK((anyShape4 == Shape{}));
    }
    {
        AnyShape<4> anyShape4{1, 2, 3, 4};
        anyShape4.demote(1);
        CHECK((anyShape4 == Shape{1}));
        anyShape4.demote(0);
        CHECK((anyShape4 == Shape{}));
    }
    {
        AnyShape<4> anyShape3{1, 2, 3};
        CHECK_THROWS(anyShape3.demote(4));
    }
    {
        AnyShape<4> anyShape3{1, 2, 3};
        CHECK_THROWS(anyShape3.demote(-1));
    }
    {
        AnyShape<4> anyShape4{1, 2, 3, 4};
        anyShape4.demote(2);
        CHECK((anyShape4 == Shape{1, 2}));
        CHECK_THROWS(anyShape4.demote(3));
    }
}

TEST_CASE("AnyShape size", "[Shape]"){
    // Explicit shapes
    STATIC_CHECK(AnyShape<5>{}.size() == 1);
    STATIC_CHECK(AnyShape<5>{5}.size() == 5);
    STATIC_CHECK(AnyShape<5>{5, 2}.size() == 10);
    STATIC_CHECK(AnyShape<5>{5, 1, 1, 2, 3}.size() == 30);

    // Implicit shapes
    STATIC_CHECK(AnyShape<5>{-1}.size() == 1);
    STATIC_CHECK(AnyShape<5>{-1, 2}.size() == 2);
    STATIC_CHECK(AnyShape<5>{2, -1}.size() == 2);
    STATIC_CHECK(AnyShape<5>{23, 4, 5, -1, 7}.size() == 3220);
}
