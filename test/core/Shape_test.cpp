//
// Created by Amy Fetzner on 11/29/2023.
//

#include "TensorII/Shape.h"
#include "Catch2/catch_test_macros.hpp"

using namespace TensorII::Core;

TEST_CASE("Shapes", "[Shape]") {
    // Valid, explicit shapes
    STATIC_CHECK(ExplicitShape_v<Shape<>>);
    STATIC_CHECK(ExplicitShape_v<Shape<5>>);
    STATIC_CHECK(ExplicitShape_v<Shape<5, 2>>);
    STATIC_CHECK(ExplicitShape_v<Shape<5, 1, 1, 3, 5>>);

    STATIC_CHECK_FALSE(ExplicitShape_v<Shape<-1>>);
    STATIC_CHECK_FALSE(ExplicitShape_v<Shape<-1, 2>>);
    STATIC_CHECK_FALSE(ExplicitShape_v<Shape<2, -1>>);
    STATIC_CHECK_FALSE(ExplicitShape_v<Shape<23, 4, -1, 5, 7>>);

    // Valid, explicit shapes
    STATIC_CHECK_FALSE(ImplicitShape_v<Shape<>>);
    STATIC_CHECK_FALSE(ImplicitShape_v<Shape<5>>);
    STATIC_CHECK_FALSE(ImplicitShape_v<Shape<5, 2>>);
    STATIC_CHECK_FALSE(ImplicitShape_v<Shape<5, 1, 1, 3, 5>>);

    STATIC_CHECK(ImplicitShape_v<Shape<-1>>);
    STATIC_CHECK(ImplicitShape_v<Shape<-1, 2>>);
    STATIC_CHECK(ImplicitShape_v<Shape<2, -1>>);
    STATIC_CHECK(ImplicitShape_v<Shape<23, 4, -1, 5, 7>>);

    //Invalid shapes
    STATIC_CHECK_FALSE(ValidShape_v<Shape<0>>);
    STATIC_CHECK_FALSE(ValidShape_v<Shape<-1, -1>>);
    STATIC_CHECK_FALSE(ValidShape_v<Shape<-1, 2, 3, -1, 4>>);
    STATIC_CHECK_FALSE(ValidShape_v<Shape<-1, 0, 1, 23, 4>>);
    STATIC_CHECK_FALSE(ValidShape_v<Shape<0, 1, 2>>);
    STATIC_CHECK_FALSE(ValidShape_v<Shape<0, 0, 0>>);
}
