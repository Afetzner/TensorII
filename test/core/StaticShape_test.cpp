//
// Created by Amy Fetzner on 11/29/2023.
//

#include "TensorII/StaticShape.h"
#include "Catch2/catch_test_macros.hpp"

using namespace TensorII::Core;

TEST_CASE("Explicit Shapes", "[Shape]") {
    // Valid, explicit shapes
    STATIC_CHECK(IsExplicitShape_v<Shape<>>);
    STATIC_CHECK(IsExplicitShape_v<Shape<5>>);
    STATIC_CHECK(IsExplicitShape_v<Shape<5, 2>>);
    STATIC_CHECK(IsExplicitShape_v<Shape<5, 1, 1, 3, 5>>);

    STATIC_CHECK_FALSE(IsExplicitShape_v<Shape<-1>>);
    STATIC_CHECK_FALSE(IsExplicitShape_v<Shape<-1, 2>>);
    STATIC_CHECK_FALSE(IsExplicitShape_v<Shape<2, -1>>);
    STATIC_CHECK_FALSE(IsExplicitShape_v<Shape<23, 4, -1, 5, 7>>);

    // Concept
    STATIC_CHECK(ExplicitShape<Shape<>>);
    STATIC_CHECK(ExplicitShape<Shape<5>>);
    STATIC_CHECK(ExplicitShape<Shape<5, 2>>);
    STATIC_CHECK(ExplicitShape<Shape<5, 1, 1, 3, 5>>);

    STATIC_CHECK_FALSE(ExplicitShape<Shape<-1>>);
    STATIC_CHECK_FALSE(ExplicitShape<Shape<-1, 2>>);
    STATIC_CHECK_FALSE(ExplicitShape<Shape<2, -1>>);
    STATIC_CHECK_FALSE(ExplicitShape<Shape<23, 4, -1, 5, 7>>);

}

TEST_CASE("Implicit Shapes", "[Shape]") {
    // Valid, implicit shapes
    STATIC_CHECK_FALSE(IsImplicitShape_v<Shape<>>);
    STATIC_CHECK_FALSE(IsImplicitShape_v<Shape<5>>);
    STATIC_CHECK_FALSE(IsImplicitShape_v<Shape<5, 2>>);
    STATIC_CHECK_FALSE(IsImplicitShape_v<Shape<5, 1, 1, 3, 5>>);

    STATIC_CHECK(IsImplicitShape_v<Shape<-1>>);
    STATIC_CHECK(IsImplicitShape_v<Shape<-1, 2>>);
    STATIC_CHECK(IsImplicitShape_v<Shape<2, -1>>);
    STATIC_CHECK(IsImplicitShape_v<Shape<23, 4, -1, 5, 7>>);

    // Concept
    STATIC_CHECK_FALSE(ImplicitShape<Shape<>>);
    STATIC_CHECK_FALSE(ImplicitShape<Shape<5>>);
    STATIC_CHECK_FALSE(ImplicitShape<Shape<5, 2>>);
    STATIC_CHECK_FALSE(ImplicitShape<Shape<5, 1, 1, 3, 5>>);

    STATIC_CHECK(ImplicitShape<Shape<-1>>);
    STATIC_CHECK(ImplicitShape<Shape<-1, 2>>);
    STATIC_CHECK(ImplicitShape<Shape<2, -1>>);
    STATIC_CHECK(ImplicitShape<Shape<23, 4, -1, 5, 7>>);
}

TEST_CASE("Invalid Shapes", "[Shape]"){
    // Invalid shapes
    STATIC_CHECK_FALSE(IsValidShape_v<Shape<0>>);
    STATIC_CHECK_FALSE(IsValidShape_v<Shape<-1, -1>>);
    STATIC_CHECK_FALSE(IsValidShape_v<Shape<-1, 2, 3, -1, 4>>);
    STATIC_CHECK_FALSE(IsValidShape_v<Shape<-1, 0, 1, 23, 4>>);
    STATIC_CHECK_FALSE(IsValidShape_v<Shape<0, 1, 2>>);
    STATIC_CHECK_FALSE(IsValidShape_v<Shape<0, 0, 0>>);

    // Concept
    STATIC_CHECK_FALSE(ValidShape<Shape<0>>);
    STATIC_CHECK_FALSE(ValidShape<Shape<-1, -1>>);
    STATIC_CHECK_FALSE(ValidShape<Shape<-1, 2, 3, -1, 4>>);
    STATIC_CHECK_FALSE(ValidShape<Shape<-1, 0, 1, 23, 4>>);
    STATIC_CHECK_FALSE(ValidShape<Shape<0, 1, 2>>);
    STATIC_CHECK_FALSE(ValidShape<Shape<0, 0, 0>>);
}

TEST_CASE("Shape deduction", "[Shape]"){
    {
        using Explicit = Shape<10>;
        using Implicit = Shape<-1>;
        using Deduced = DeduceShape<Explicit, Implicit>::Shape;
        STATIC_CHECK(std::is_same_v<Deduced, Shape<10>>);
    }
    {
        using Explicit = Shape<2, 3, 5>;
        using Implicit = Shape<5, -1>;
        using Deduced = DeduceShape<Explicit, Implicit>::Shape;
        STATIC_CHECK(std::is_same_v<Deduced, Shape<5, 6>>);
    }
    {
        using Explicit = Shape<2, 2, 6, 7, 1>;
        using Implicit = Shape<1, -1, 21>;
        using Deduced = DeduceShape<Explicit, Implicit>::Shape;
        STATIC_CHECK(std::is_same_v<Deduced, Shape<1, 8, 21>>);
    }
    {
        using Explicit = Shape<10>;
        using Implicit = Shape<3, -1>;
        STATIC_CHECK(!DeduceShape<Explicit, Implicit>::deducible);
    }
}
