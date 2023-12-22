//
// Created by Amy Fetzner on 11/29/2023.
//

#include "TensorII/Shape.h"
#include "Catch2/catch_test_macros.hpp"

using namespace TensorII::Core;

TEST_CASE("Explicit Dynamic Shapes", "[Shape]") {
    // Valid, explicit shapes
    STATIC_CHECK(Shape{}.isValidExplicit());
    STATIC_CHECK(Shape{5}.isValidExplicit());
    STATIC_CHECK(Shape{5, 2}.isValidExplicit());
    STATIC_CHECK(Shape{5, 1, 1, 2, 3}.isValidExplicit());

    // valid implicit shapes
    STATIC_CHECK_FALSE(Shape{-1}.isValidExplicit());
    STATIC_CHECK_FALSE(Shape{-1, 2}.isValidExplicit());
    STATIC_CHECK_FALSE(Shape{2, -1}.isValidExplicit());
    STATIC_CHECK_FALSE(Shape{23, 4, 5, -1, 7}.isValidExplicit());

    // Invalid shapes
    STATIC_CHECK_FALSE(Shape{1, 0, 3}.isValidExplicit());           // invalid, 0
    STATIC_CHECK_FALSE(Shape{4, -1, 2, -1, 3}.isValidExplicit());   // invalid, two -1s
}

TEST_CASE("Implicit Shapes", "[Shape]") {
    // Valid, explicit shapes
    STATIC_CHECK_FALSE(Shape{}.isValidImplicit());
    STATIC_CHECK_FALSE(Shape{5}.isValidImplicit());
    STATIC_CHECK_FALSE(Shape{5, 2}.isValidImplicit());
    STATIC_CHECK_FALSE(Shape{5, 1, 1, 2, 3}.isValidImplicit());

    // valid implicit shapes
    STATIC_CHECK(Shape{-1}.isValidImplicit());
    STATIC_CHECK(Shape{-1, 2}.isValidImplicit());
    STATIC_CHECK(Shape{2, -1}.isValidImplicit());
    STATIC_CHECK(Shape{23, 4, 5, -1, 7}.isValidImplicit());

    // Invalid shapes
    STATIC_CHECK_FALSE(Shape{1, 0, 3}.isValidImplicit());           // invalid, 0
    STATIC_CHECK_FALSE(Shape{4, -1, 2, -1, 3}.isValidImplicit());   // invalid, two -1s
}

TEST_CASE("Invalid Shapes", "[Shape]"){
    // Invalid shapes
    STATIC_CHECK_FALSE(Shape{0}.isValid());
    STATIC_CHECK_FALSE(Shape{-1, -1}.isValid());
    STATIC_CHECK_FALSE(Shape{-1, 2, 3, -1, 4}.isValid());
    STATIC_CHECK_FALSE(Shape{-1, 0, 1, 23, 4}.isValid());
    STATIC_CHECK_FALSE(Shape{0, 0, 0}.isValid());
}

TEST_CASE("Shape construction, Args", "[Shape]"){
    constexpr Shape<0> shape0 = Shape();
    static_assert(shape0 == Shape{});

    constexpr Shape<1> shape1 = Shape(1);
    static_assert(shape1 == Shape{1});

    constexpr Shape<2> shape2 = Shape(1, 2);
    static_assert(shape2 == Shape{1, 2});

    constexpr Shape<3> shape3 = Shape(1, 2, 3);
    static_assert(shape3 == Shape{1, 2, 3});

    constexpr Shape<4> shape4 = Shape(1, 2, 3, 4);
    static_assert(shape4 == Shape{1, 2, 3, 4});
}

TEST_CASE("Shape construction, std::array", "[Shape]"){
    constexpr std::array<tensorDimension, 1> arr1 {1};
    constexpr Shape<1> shape1 = Shape<1>(from_range, arr1);
    static_assert(shape1 == Shape{1});

    constexpr std::array<tensorDimension, 2> arr2 {1, 2};
    constexpr Shape<2> shape2 = Shape<2>(from_range, arr2);
    static_assert(shape2 == Shape{1, 2});

    constexpr std::array<tensorDimension, 3> arr3 {1, 2, 3};
    constexpr Shape<3> shape3 = Shape<3>(from_range, arr3);
    static_assert(shape3 == Shape{1, 2, 3});

    constexpr std::array<tensorDimension, 4> arr4 {1, 2, 3, 4};
    constexpr Shape<4> shape4 = Shape<4>(from_range, arr4);
    static_assert(shape4 == Shape{1, 2, 3, 4});
}

TEST_CASE("Shape n_elems", "[Shape]"){
    // Explicit shapes
    STATIC_CHECK(Shape{}.n_elems() == 1);
    STATIC_CHECK(Shape{5}.n_elems() == 5);
    STATIC_CHECK(Shape{5, 2}.n_elems() == 10);
    STATIC_CHECK(Shape{5, 1, 1, 2, 3}.n_elems() == 30);

    // Implicit shapes
    STATIC_CHECK(Shape{-1}.n_elems() == 1);
    STATIC_CHECK(Shape{-1, 2}.n_elems() == 2);
    STATIC_CHECK(Shape{2, -1}.n_elems() == 2);
    STATIC_CHECK(Shape{23, 4, 5, -1, 7}.n_elems() == 3220);
}

TEST_CASE("Shape deduction", "[Shape]"){
    {
        constexpr Shape Explicit = Shape{10};
        constexpr Shape Implicit = Shape{-1};
        constexpr Shape Deduced = deduceShape(Explicit, Implicit);
        STATIC_CHECK(Deduced == Shape{10});
    }
    {
        constexpr Shape Explicit = Shape{2, 3, 5};
        constexpr Shape Implicit = Shape{5, -1};
        constexpr Shape Deduced = deduceShape(Explicit, Implicit);
        STATIC_CHECK(Deduced == Shape{5, 6});
    }
    {
        constexpr Shape Explicit = Shape{2, 2, 6, 7, 1};
        constexpr Shape Implicit = Shape{1, -1, 21};
        constexpr Shape Deduced = deduceShape(Explicit, Implicit);
        STATIC_CHECK(Deduced == Shape{1, 8, 21});
    }

    // Incompatible shapes
    {
        constexpr Shape Explicit = Shape{10};
        constexpr Shape Implicit = Shape{3, -1};
        REQUIRE_THROWS(deduceShape(Explicit, Implicit));
    }

    // Using explicit shape as implicit
    {
        constexpr Shape Explicit = Shape{10};
        constexpr Shape Implicit = Shape{10};
        REQUIRE_THROWS(deduceShape(Explicit, Implicit));
        // Need to find a way to verify this fails to compile if constexpr.
//        constexpr Shape Deduced = DeduceShape(Explicit, Implicit);
    }
    // using implicit shape as explicit
    {
        constexpr Shape Explicit = Shape{3, -1};
        constexpr Shape Implicit = Shape{3, -1};
        REQUIRE_THROWS(deduceShape(Explicit, Implicit));
        // Need to find a way to verify this fails to compile if constexpr.
//        constexpr Shape Deduced = DeduceShape(Explicit, Implicit);
    }
}

TEST_CASE("Shape augment args", "[Shape]"){
    {
        constexpr Shape before = Shape {1, 2, 3};
        constexpr Shape after = before.augmented(4);
        static_assert(after == Shape{1, 2, 3, 4});
    }
    {
        constexpr Shape before = Shape {1, 2, 3};
        constexpr Shape after = before.augmented(4, 5, 6, 7);
        static_assert(after == Shape{1, 2, 3, 4, 5, 6, 7});
    }
    {
        constexpr Shape before = Shape {};
        constexpr Shape after = before.augmented(1, 2, 3);
        static_assert(after == Shape{1, 2, 3});
    }
}

TEST_CASE("Shape augment array", "[Shape]"){
    {
        constexpr Shape before = Shape {1, 2, 3};
        constexpr Shape after = before.augmented<4>(from_range, std::array{4});
        static_assert(after == Shape{1, 2, 3, 4});
    }
    {
        constexpr Shape before = Shape {1, 2, 3};
        constexpr Shape after = before.augmented<7>(from_range, std::array{4, 5, 6, 7});
        static_assert(after == Shape{1, 2, 3, 4, 5, 6, 7});
    }
    {
        constexpr Shape before = Shape {};
        constexpr Shape after = before.augmented<3>(from_range, std::array{1, 2, 3});
        static_assert(after == Shape{1, 2, 3});
    }
}

TEST_CASE("Shape demote", "[Shape]"){
    {
        constexpr Shape before = Shape {1, 2, 3};
        constexpr Shape after = before.demoted<2>();
        static_assert(after == Shape{1, 2});
    }
    {
        constexpr Shape before = Shape {1, 2, 3};
        constexpr Shape after = before.demoted<0>();
        static_assert(after == Shape{});
    }
}
