//
// Created by Amy Fetzner on 12/10/2023.
//

// NOTE: All equality comparisons inside Catch2 checks in this file must be wrapped in parentheses
// STATIC_CHECK(anyShape0 == Shape{});  -->  STATIC_CHECK((anyShape0 == Shape{}));
// because shapes are ranges, Catch2 is trying to use its own range comparator instead of the defined operator==

#include "TensorII/AnyShape.h"
#include "Catch2/catch_test_macros.hpp"
#include "Catch2/catch_template_test_macros.hpp"

using namespace TensorII::Core;

TEST_CASE("AnyShape construction", "[AnyShape]") {
    SECTION("From args", "AnyShapes created with array initialization should equal the same crated by passing args") {
        AnyShape<4> anyShape0 = {};
        CHECK(anyShape0 == Shape{});

        AnyShape<4> anyShape1 = {1};
        CHECK(anyShape1 == Shape{1});

        AnyShape<4> anyShape2 = {1, 2};
        CHECK(anyShape2 == Shape{1, 2});

        AnyShape<4> anyShape3 = {1, 2, 3};
        CHECK(anyShape3 == Shape{1, 2, 3});

        AnyShape<4> anyShape4 = {1, 2, 3, 4};
        CHECK(anyShape4 == Shape{1, 2, 3, 4});
    }
    SECTION("From static shape", "AnyShapes created with array initialization should equal the same set from static shape") {
        AnyShape<4> anyShape0 = Shape{};
        CHECK(anyShape0 == Shape{});

        AnyShape<4> anyShape1 = Shape{1};
        CHECK(anyShape1 == Shape{1});

        AnyShape<4> anyShape2 = Shape{1, 2};
        CHECK(anyShape2 == Shape{1, 2});

        AnyShape<4> anyShape3 = Shape{1, 2, 3};
        CHECK(anyShape3 == Shape{1, 2, 3});

        AnyShape<4> anyShape4 = Shape{1, 2, 3, 4};
        CHECK(anyShape4 == Shape{1, 2, 3, 4});
    }
}

TEMPLATE_TEST_CASE("AnyShape construction from range", "[AnyShape][template]",
                   (std::array<int, 4>), (std::array<size_t, 4>), std::vector<int>, std::vector<size_t>) {
    GIVEN("A range of 4 values"){
            TestType range = {1, 2, 3, 4};

        THEN("The range has n_elems 4"){
            REQUIRE(std::ranges::size(range) == 4);
        }

        WHEN("An AnyShape is created from that range"){
            AnyShape<8> anyShape = AnyShape<8>{from_range, range};

            THEN("The AnyShape also has n_elems 4"){
                CHECK(anyShape.rank() == 4);
                CHECK(std::ranges::size(anyShape) == 4);
            }

            THEN("The AnyShape has the same values") {
                CHECK(anyShape[0] == 1);
                CHECK(anyShape[1] == 2);
                CHECK(anyShape[2] == 3);
                CHECK(anyShape[3] == 4);
            }
        }

        WHEN("An AnyShape smaller than that range is created from that range") {
            THEN("It throws"){
                CHECK_THROWS(AnyShape<3>{from_range, range});
            }
        }
    }
}

SCENARIO("AnyShape can have variadic arguments emplaced", "[AnyShape]"){
    GIVEN("An anyshape of size 4"){
        AnyShape<4> anyShape;

        WHEN("0 arguments are emplaced into it"){
            anyShape.emplace();
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 0);
                CHECK((anyShape == Shape{}));
            }
        }
        WHEN("1 argument is emplaced into it"){
            anyShape.emplace(1);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 1);
                CHECK((anyShape == Shape{1}));
            }
        }
        WHEN("2 arguments are emplaced into it"){
            anyShape.emplace(1, 2);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 2);
                CHECK((anyShape == Shape{1, 2}));
            }
        }
        WHEN("3 arguments are emplaced into it"){
            anyShape.emplace(1, 2, 3);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 3);
                CHECK((anyShape == Shape{1, 2, 3}));
            }
        }
        WHEN("4 arguments are emplaced into it"){
            anyShape.emplace(1, 2, 3, 4);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 4);
                CHECK((anyShape == Shape{1, 2, 3, 4}));
            }
        }
    }
}

TEMPLATE_TEST_CASE("AnyShape emplacement from range", "[AnyShape][template]",
                   (std::array<int, 4>), (std::array<size_t, 4>), std::vector<int>, std::vector<size_t>) {
    GIVEN("A range of 4 values"){
        TestType range = {1, 2, 3, 4};

        THEN("The range has n_elems 4"){
            REQUIRE(std::ranges::size(range) == 4);
        }

        WHEN("A range is emplaced into that AnyShape"){
            AnyShape<8> anyShape {100, 200, 300, 400, 500, 600};
            anyShape.emplace(from_range, range);

            THEN("The AnyShape now also has n_elems 4"){
                CHECK(anyShape.rank() == 4);
                CHECK(std::ranges::size(anyShape) == 4);
            }

            THEN("The AnyShape has the same values") {
                CHECK(anyShape[0] == 1);
                CHECK(anyShape[1] == 2);
                CHECK(anyShape[2] == 3);
                CHECK(anyShape[3] == 4);
            }
        }

        WHEN("An AnyShape smaller than that range is created from that range") {
            AnyShape<3> anyShape {100, 200};

            THEN("It throws"){
                CHECK_THROWS(anyShape.emplace(from_range, range));
            }
        }
    }
}

SCENARIO("AnyShape can create augmented with variadic arguments", "[AnyShape]"){
    GIVEN("An anyshape of size 8, with rank 4"){
        AnyShape<8> smallAnyShape {1, 2, 3, 4};
        THEN("It has rank 4") {
            REQUIRE(smallAnyShape.rank() == 4);
        }

        WHEN("0 arguments are augmented into it"){
            auto anyShape = smallAnyShape.augmented();
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 4);
                CHECK((anyShape == Shape{1, 2, 3, 4}));
            }
        }
        WHEN("1 argument is augmented into it"){
            auto anyShape = smallAnyShape.augmented(5);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 5);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5}));
            }
        }
        WHEN("2 arguments are augmented into it"){
            auto anyShape = smallAnyShape.augmented(5, 6);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 6);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5, 6}));
            }
        }
        WHEN("3 arguments are augmented into it"){
            auto anyShape = smallAnyShape.augmented(5, 6, 7);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 7);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5, 6, 7}));
            }
        }
        WHEN("4 arguments are augmented into it"){
            auto anyShape = smallAnyShape.augmented(5, 6, 7, 8);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 8);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5, 6, 7, 8}));
            }
        }
        WHEN("5 arguments are augmented into it"){
            THEN ("It throws"){
                CHECK_THROWS(smallAnyShape.augmented(5, 6, 7, 8, 9));
            }
        }
    }
}

SCENARIO("AnyShape can be augmented by variadic arguments", "[AnyShape]"){
    GIVEN("An anyshape of size 8, with rank 4"){
        AnyShape<8> anyShape {1, 2, 3, 4};

        THEN("The anyShape has rank 4"){
            REQUIRE(anyShape.rank() == 4);
        }

        WHEN("0 arguments are emplaced into it"){
            anyShape.augment();
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 4);
                CHECK((anyShape == Shape{1, 2, 3, 4}));
            }
        }
        WHEN("1 argument is emplaced into it"){
            anyShape.augment(5);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 5);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5}));
            }
        }
        WHEN("2 arguments are emplaced into it"){
            anyShape.augment(5, 6);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 6);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5, 6}));
            }
        }
        WHEN("3 arguments are emplaced into it"){
            anyShape.augment(5, 6, 7);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 7);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5, 6, 7}));
            }
        }
        WHEN("4 arguments are emplaced into it"){
            anyShape.augment(5, 6, 7, 8);
            THEN ("It has those dimensions and rank"){
                CHECK(anyShape.rank() == 8);
                CHECK((anyShape == Shape{1, 2, 3, 4, 5, 6, 7, 8}));
            }
        }
        WHEN("5 arguments are emplaced into it"){
            THEN ("It throws"){
                CHECK_THROWS(anyShape.augment(5, 6, 7, 8, 9));
            }
        }
    }
}

TEMPLATE_TEST_CASE("AnyShape create augmented from a range", "[AnyShape][template]",
                   (std::array<int, 4>), (std::array<size_t, 4>), std::vector<int>, std::vector<size_t>) {
    GIVEN("A shape of rank 4 and a range of 4 values"){
        AnyShape<8> smallAnyShape {1, 2, 3, 4};
        TestType range = {5, 6, 7, 8};

        THEN("The range has n_elems 4 and the shape has rank 4"){
            REQUIRE(std::ranges::size(range) == 4);
            REQUIRE(smallAnyShape.rank() == 4);
        }

        WHEN("The anyShape is augmented") {
            auto anyShape = smallAnyShape.augmented(from_range, range);
            THEN("The shape has the right rank and dimensions"){
                CHECK(anyShape.rank() == 8);
                CHECK(anyShape == Shape{1, 2, 3, 4, 5, 6, 7, 8});
            }
        }
    }

    GIVEN("A shape of rank 6 and a range of 4 values too large for that AnyShape"){
        AnyShape<8> smallAnyShape {1, 2, 3, 4, 6, 7};
        TestType range = {5, 6, 7, 8};

        THEN("The range has n_elems 4 and the shape has rank 6"){
            REQUIRE(std::ranges::size(range) == 4);
            REQUIRE(smallAnyShape.rank() == 6);
        }

        WHEN("The anyShape is augmented") {
            THEN("It throws"){
                CHECK_THROWS(smallAnyShape.augmented(from_range, range));
            }
        }
    }
}

TEMPLATE_TEST_CASE("AnyShape augment from a range", "[AnyShape][template]",
                   (std::array<int, 4>), (std::array<size_t, 4>), std::vector<int>, std::vector<size_t>) {
    GIVEN("A shape of rank 4 and a range of 4 values"){
        AnyShape<8> anyShape {1, 2, 3, 4};
        TestType range = {5, 6, 7, 8};

        THEN("The range has n_elems 4 and the shape has rank 4"){
            REQUIRE(std::ranges::size(range) == 4);
            REQUIRE(anyShape.rank() == 4);
        }

        WHEN("The anyShape is augmented") {
            anyShape.augment(from_range, range);
            THEN("The shape has the right rank and dimensions"){
                CHECK(anyShape.rank() == 8);
                CHECK(anyShape == Shape{1, 2, 3, 4, 5, 6, 7, 8});
            }
        }
    }

    GIVEN("A shape of rank 6 and a range of 4 values too large for that AnyShape"){
        AnyShape<8> anyShape {1, 2, 3, 4, 6, 7};
        TestType range = {5, 6, 7, 8};

        THEN("The range has n_elems 4 and the shape has rank 6"){
            REQUIRE(std::ranges::size(range) == 4);
            REQUIRE(anyShape.rank() == 6);
        }

        WHEN("The anyShape is augmented") {
            THEN("It throws"){
                CHECK_THROWS(anyShape.augment(from_range, range));
            }
        }
    }
}

SCENARIO("AnyShape can create a demoted AnyShape", "[AnyShape]") {
    GIVEN("An anyshape with 4 dimensions"){
        AnyShape<8> largeAnyShape{1, 2, 3, 4};

        THEN("The AnyShape has rank 4") {
            REQUIRE(largeAnyShape.rank() == 4);
        }

        WHEN("The shape is demoted to rank 4"){
            auto anyShape = largeAnyShape.demoted(4);
            THEN ("The new shape has rank 4 and the same values") {
                CHECK(anyShape.rank() == 4);
                CHECK(anyShape == Shape{1, 2, 3, 4});
            }
        }
        WHEN("The shape is demoted to rank 3"){
            auto anyShape = largeAnyShape.demoted(3);
            THEN ("The new shape has rank 3 and the same values") {
                CHECK(anyShape.rank() == 3);
                CHECK(anyShape == Shape{1, 2, 3});
            }
        }
        WHEN("The shape is demoted to rank 2"){
            auto anyShape = largeAnyShape.demoted(2);
            THEN ("The new shape has rank 2 and the same values") {
                CHECK(anyShape.rank() == 2);
                CHECK(anyShape == Shape{1, 2});
            }
        }
        WHEN("The shape is demoted to rank 1"){
            auto anyShape = largeAnyShape.demoted(1);
            THEN ("The new shape has rank 1 and the same values") {
                CHECK(anyShape.rank() == 1);
                CHECK(anyShape == Shape{1});
            }
        }
        WHEN("The shape is demoted to rank 0"){
            auto anyShape = largeAnyShape.demoted(0);
            THEN ("The new shape has rank 0 and the same values") {
                CHECK(anyShape.rank() == 0);
                CHECK(anyShape == Shape{});
            }
        }
        WHEN("The shape is demoted to rank 5"){
            THEN ("It throws") {
                CHECK_THROWS(largeAnyShape.demoted(5));
            }
        }
    }
}

SCENARIO("AnyShape can be demoted", "[AnyShape]") {
    GIVEN("An anyshape with 4 dimensions"){
        AnyShape<8> anyShape{1, 2, 3, 4};

        THEN("The AnyShape has rank 4") {
            REQUIRE(anyShape.rank() == 4);
        }

        WHEN("The shape is demoted to rank 4"){
            anyShape.demote(4);
            THEN ("The new shape has rank 4 and the same values") {
                CHECK(anyShape.rank() == 4);
                CHECK(anyShape == Shape{1, 2, 3, 4});
            }
        }
        WHEN("The shape is demoted to rank 3"){
            anyShape.demote(3);
            THEN ("The new shape has rank 3 and the same values") {
                CHECK(anyShape.rank() == 3);
                CHECK(anyShape == Shape{1, 2, 3});
            }
        }
        WHEN("The shape is demoted to rank 2"){
            anyShape.demote(2);
            THEN ("The new shape has rank 2 and the same values") {
                CHECK(anyShape.rank() == 2);
                CHECK(anyShape == Shape{1, 2});
            }
        }
        WHEN("The shape is demoted to rank 1"){
            anyShape.demote(1);
            THEN ("The new shape has rank 1 and the same values") {
                CHECK(anyShape.rank() == 1);
                CHECK(anyShape == Shape{1});
            }
        }
        WHEN("The shape is demoted to rank 0"){
            anyShape.demote(0);
            THEN ("The new shape has rank 0 and the same values") {
                CHECK(anyShape.rank() == 0);
                CHECK(anyShape == Shape{});
            }
        }
        WHEN("The shape is demoted to rank 5"){
            THEN ("It throws") {
                CHECK_THROWS(anyShape.demote(5));
            }
        }
    }
}

TEST_CASE("AnyShape n_elems", "[AnyShape][static]"){
    SECTION("Explicit shapes"){
        STATIC_CHECK(AnyShape<6>{}.n_elems() == 1);
        STATIC_CHECK(AnyShape<6>{5}.n_elems() == 5);
        STATIC_CHECK(AnyShape<6>{5, 2}.n_elems() == 10);
        STATIC_CHECK(AnyShape<6>{5, 1, 1, 2, 3}.n_elems() == 30);
    }

    SECTION("Implicit shapes"){
        STATIC_CHECK(AnyShape<6>{-1}.n_elems() == 1);
        STATIC_CHECK(AnyShape<6>{-1, 2}.n_elems() == 2);
        STATIC_CHECK(AnyShape<6>{2, -1}.n_elems() == 2);
        STATIC_CHECK(AnyShape<6>{23, 4, 5, -1, 7}.n_elems() == 3220);
    }
}

TEST_CASE("AnyShape assignment from static shape", "[AnyShape]") {
    GIVEN("An AnyShape"){
        AnyShape<4> anyShape {100, 200, 300};

        WHEN("It is assigned to a rank 0 shape"){
            anyShape = Shape{};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 0);
                CHECK(anyShape == Shape{});
            }
        }
        WHEN("It is assigned to a rank 1 shape"){
            anyShape = Shape{1};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 1);
                CHECK(anyShape == Shape{1});
            }
        }
        WHEN("It is assigned to a rank 2 shape"){
            anyShape = Shape{1, 2};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 2);
                CHECK(anyShape == Shape{1, 2});
            }
        }
        WHEN("It is assigned to a rank 3 shape"){
            anyShape = Shape{1, 2, 3};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 3);
                CHECK(anyShape == Shape{1, 2, 3});
            }
        }
        WHEN("It is assigned to a rank 4 shape"){
            anyShape = Shape{1, 2, 3, 4};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 4);
                CHECK(anyShape == Shape{1, 2, 3, 4});
            }
        }
    }
}

TEST_CASE("AnyShape assignment from AnyShape", "[AnyShape]") {
    GIVEN("An AnyShape"){
        AnyShape<4> anyShape {100, 200, 300};

        WHEN("It is assigned to a rank 0 AnyShape"){
            anyShape = AnyShape<4>{};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 0);
                CHECK(anyShape == Shape{});
            }
        }
        WHEN("It is assigned to a rank 1 AnyShape"){
            anyShape = AnyShape<4>{1};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 1);
                CHECK(anyShape == Shape{1});
            }
        }
        WHEN("It is assigned to a rank 2 AnyShape"){
            anyShape = AnyShape<4>{1, 2};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 2);
                CHECK(anyShape == Shape{1, 2});
            }
        }
        WHEN("It is assigned to a rank 3 AnyShape"){
            anyShape = AnyShape<4>{1, 2, 3};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 3);
                CHECK(anyShape == Shape{1, 2, 3});
            }
        }
        WHEN("It is assigned to a rank 4 AnyShape"){
            anyShape = AnyShape<4>{1, 2, 3, 4};
            THEN("It has the correct rank and dimensions"){
                CHECK(anyShape.rank() == 4);
                CHECK(anyShape == Shape{1, 2, 3, 4});
            }
        }
    }
}

TEST_CASE("AnyShape equality to AnyShape", "[AnyShape]") {
    CHECK(AnyShape<4>{} == AnyShape<4>{});
    CHECK(AnyShape<4>{1, 2} == AnyShape<4>{1, 2});
    CHECK(AnyShape<4>{1, 2, 3, 4} == AnyShape<4>{1, 2, 3, 4});

    CHECK(AnyShape<4>{} == AnyShape<8>{});
    CHECK(AnyShape<8>{} == AnyShape<4>{});

    CHECK(AnyShape<4>{1, 2} == AnyShape<8>{1, 2});
    CHECK(AnyShape<8>{1, 2} == AnyShape<4>{1, 2});

    CHECK(AnyShape<4>{1, 2, 3, 4} == AnyShape<8>{1, 2, 3, 4});
    CHECK(AnyShape<8>{1, 2, 3, 4} == AnyShape<4>{1, 2, 3, 4});
}

TEST_CASE("AnyShape equality to static shape", "[AnyShape]") {
    CHECK(AnyShape<4>{} == Shape<0>{});
    CHECK(AnyShape<4>{1, 2} == Shape<2>{1, 2});
    CHECK(AnyShape<4>{1, 2, 3, 4} == Shape<4>{1, 2, 3, 4});

    CHECK(AnyShape<4>{} == Shape<0>{});
    CHECK(AnyShape<4>{1, 2} == Shape<2>{1, 2});
    CHECK(AnyShape<4>{1, 2, 3, 4} == Shape<4>{1, 2, 3, 4});
}

TEST_CASE("AnyShape validity checks", "[AnyShape]") {
    SECTION("Explicit AnyShapes"){
        // Explicit shapes are valid explicit shapes
        STATIC_CHECK(AnyShape<8>{}.isValidExplicit());
        STATIC_CHECK(AnyShape<8>{5}.isValidExplicit());
        STATIC_CHECK(AnyShape<8>{5, 2}.isValidExplicit());
        STATIC_CHECK(AnyShape<8>{5, 1, 1, 2, 3}.isValidExplicit());

        // Explicit shapes are not valid implicit shapes
        STATIC_CHECK_FALSE(AnyShape<8>{}.isValidImplicit());
        STATIC_CHECK_FALSE(AnyShape<8>{5}.isValidImplicit());
        STATIC_CHECK_FALSE(AnyShape<8>{5, 2}.isValidImplicit());
        STATIC_CHECK_FALSE(AnyShape<8>{5, 1, 1, 2, 3}.isValidImplicit());
    }

    SECTION("Implicit AnyShapes") {
        // Implicit shapes are valid implicit shapes
        STATIC_CHECK(AnyShape<8>{-1}.isValidImplicit());
        STATIC_CHECK(AnyShape<8>{-1, 2}.isValidImplicit());
        STATIC_CHECK(AnyShape<8>{2, -1}.isValidImplicit());
        STATIC_CHECK(AnyShape<8>{23, 4, 5, -1, 7}.isValidImplicit());

        // Implicit shapes are not valid explicit shapes
        STATIC_CHECK_FALSE(AnyShape<8>{-1}.isValidExplicit());
        STATIC_CHECK_FALSE(AnyShape<8>{-1, 2}.isValidExplicit());
        STATIC_CHECK_FALSE(AnyShape<8>{2, -1}.isValidExplicit());
        STATIC_CHECK_FALSE(AnyShape<8>{23, 4, 5, -1, 7}.isValidExplicit());
    }

    SECTION("Invalid AnyShapes"){
        // Invalid shapes are not valid explicit
        STATIC_CHECK_FALSE(AnyShape<8>{1, 0, 3}.isValidExplicit());           // contains 0
        STATIC_CHECK_FALSE(AnyShape<8>{0}.isValidExplicit());                 // contains 0
        STATIC_CHECK_FALSE(AnyShape<8>{0, 0, 0}.isValidExplicit());           // contains many 0s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, -1}.isValidExplicit());            // contains multiple -1s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, 2, 3, -1, 4}.isValidExplicit());   // contains multiple -1s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, 0, 1, 23, 4}.isValidExplicit());   // contains -1 and 0

        // Invalid shapes are not valid implicit
        STATIC_CHECK_FALSE(AnyShape<8>{1, 0, 3}.isValidImplicit());           // contains 0
        STATIC_CHECK_FALSE(AnyShape<8>{0}.isValidImplicit());                 // contains 0
        STATIC_CHECK_FALSE(AnyShape<8>{0, 0, 0}.isValidImplicit());           // contains many 0s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, -1}.isValidImplicit());            // contains multiple -1s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, 2, 3, -1, 4}.isValidImplicit());   // contains multiple -1s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, 0, 1, 23, 4}.isValidImplicit());   // contains -1 and 0

        // Invalid shapes are not valid at all
        STATIC_CHECK_FALSE(AnyShape<8>{1, 0, 3}.isValid());                   // contains 0
        STATIC_CHECK_FALSE(AnyShape<8>{0}.isValid());                         // contains 0
        STATIC_CHECK_FALSE(AnyShape<8>{0, 0, 0}.isValid());                   // contains many 0s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, -1}.isValid());                    // contains multiple -1s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, 2, 3, -1, 4}.isValid());           // contains multiple -1s
        STATIC_CHECK_FALSE(AnyShape<8>{-1, 0, 1, 23, 4}.isValid());           // contains -1 and 0
    }
}
