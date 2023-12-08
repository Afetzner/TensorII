//
// Created by Amy Fetzner on 11/29/2023.
//

#include "TensorII/DynamicShape.h"
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

TEST_CASE("Shape size", "[Shape]"){
    // Explicit shapes
    STATIC_CHECK(Shape{}.size() == 1);
    STATIC_CHECK(Shape{5}.size() == 5);
    STATIC_CHECK(Shape{5, 2}.size() == 10);
    STATIC_CHECK(Shape{5, 1, 1, 2, 3}.size() == 30);

    // Implicit shapes
    STATIC_CHECK(Shape{-1}.size() == 1);
    STATIC_CHECK(Shape{-1, 2}.size() == 2);
    STATIC_CHECK(Shape{2, -1}.size() == 2);
    STATIC_CHECK(Shape{23, 4, 5, -1, 7}.size() == 3220);
}

TEST_CASE("Shape deduction", "[Shape]"){
    {
        constexpr Shape Explicit = Shape{10};
        constexpr Shape Implicit = Shape{-1};
        constexpr Shape Deduced = DeduceShape(Explicit, Implicit);
        STATIC_CHECK(Deduced == Shape{10});
    }
    {
        constexpr Shape Explicit = Shape{2, 3, 5};
        constexpr Shape Implicit = Shape{5, -1};
        constexpr Shape Deduced = DeduceShape(Explicit, Implicit);
        STATIC_CHECK(Deduced == Shape{5, 6});
    }
    {
        constexpr Shape Explicit = Shape{2, 2, 6, 7, 1};
        constexpr Shape Implicit = Shape{1, -1, 21};
        constexpr Shape Deduced = DeduceShape(Explicit, Implicit);
        STATIC_CHECK(Deduced == Shape{1, 8, 21});
    }

    // Incompatible shapes
    {
        constexpr Shape Explicit = Shape{10};
        constexpr Shape Implicit = Shape{3, -1};
        REQUIRE_THROWS(DeduceShape(Explicit, Implicit));
    }

    // Using explicit shape as implicit
    {
        constexpr Shape Explicit = Shape{10};
        constexpr Shape Implicit = Shape{10};
        REQUIRE_THROWS(DeduceShape(Explicit, Implicit));
        // Need to find a way to verify this fails to compile if constexpr
//        constexpr Shape Deduced = DeduceShape(Explicit, Implicit);
    }
    // using implicit shape as explicit
    {
        constexpr Shape Explicit = Shape{3, -1};
        constexpr Shape Implicit = Shape{3, -1};
        REQUIRE_THROWS(DeduceShape(Explicit, Implicit));
        // Need to find a way to verify this fails to compile if constexpr
//        constexpr Shape Deduced = DeduceShape(Explicit, Implicit);
    }
}
