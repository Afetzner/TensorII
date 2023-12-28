//
// Created by Amy Fetzner on 11/29/2023.
//

#include "TensorII/Shape.h"
#include "Catch2/catch_test_macros.hpp"
#include "Catch2/catch_template_test_macros.hpp"

using namespace TensorII::Core;

TEST_CASE("Shape validity checks", "[Shape][static]") {
    SECTION("Explicit shapes"){
        // Explicit shapes are valid explicit shapes
        STATIC_CHECK(Shape{}.isValidExplicit());
        STATIC_CHECK(Shape{5}.isValidExplicit());
        STATIC_CHECK(Shape{5, 2}.isValidExplicit());
        STATIC_CHECK(Shape{5, 1, 1, 2, 3}.isValidExplicit());

        // Explicit shapes are not valid implicit shapes
        STATIC_CHECK_FALSE(Shape{}.isValidImplicit());
        STATIC_CHECK_FALSE(Shape{5}.isValidImplicit());
        STATIC_CHECK_FALSE(Shape{5, 2}.isValidImplicit());
        STATIC_CHECK_FALSE(Shape{5, 1, 1, 2, 3}.isValidImplicit());
    }

    SECTION("Implicit shapes") {
        // Implicit shapes are valid implicit shapes
        STATIC_CHECK(Shape{-1}.isValidImplicit());
        STATIC_CHECK(Shape{-1, 2}.isValidImplicit());
        STATIC_CHECK(Shape{2, -1}.isValidImplicit());
        STATIC_CHECK(Shape{23, 4, 5, -1, 7}.isValidImplicit());

        // Implicit shapes are not valid explicit shapes
        STATIC_CHECK_FALSE(Shape{-1}.isValidExplicit());
        STATIC_CHECK_FALSE(Shape{-1, 2}.isValidExplicit());
        STATIC_CHECK_FALSE(Shape{2, -1}.isValidExplicit());
        STATIC_CHECK_FALSE(Shape{23, 4, 5, -1, 7}.isValidExplicit());
    }

    SECTION("Invalid shapes"){
        // Invalid shapes are not valid explicit
        STATIC_CHECK_FALSE(Shape{1, 0, 3}.isValidExplicit());           // contains 0
        STATIC_CHECK_FALSE(Shape{0}.isValidExplicit());                 // contains 0
        STATIC_CHECK_FALSE(Shape{0, 0, 0}.isValidExplicit());           // contains many 0s
        STATIC_CHECK_FALSE(Shape{-1, -1}.isValidExplicit());            // contains multiple -1s
        STATIC_CHECK_FALSE(Shape{-1, 2, 3, -1, 4}.isValidExplicit());   // contains multiple -1s
        STATIC_CHECK_FALSE(Shape{-1, 0, 1, 23, 4}.isValidExplicit());   // contains -1 and 0

        // Invalid shapes are not valid implicit
        STATIC_CHECK_FALSE(Shape{1, 0, 3}.isValidImplicit());           // contains 0
        STATIC_CHECK_FALSE(Shape{0}.isValidImplicit());                 // contains 0
        STATIC_CHECK_FALSE(Shape{0, 0, 0}.isValidImplicit());           // contains many 0s
        STATIC_CHECK_FALSE(Shape{-1, -1}.isValidImplicit());            // contains multiple -1s
        STATIC_CHECK_FALSE(Shape{-1, 2, 3, -1, 4}.isValidImplicit());   // contains multiple -1s
        STATIC_CHECK_FALSE(Shape{-1, 0, 1, 23, 4}.isValidImplicit());   // contains -1 and 0

        // Invalid shapes are not valid at all
        STATIC_CHECK_FALSE(Shape{1, 0, 3}.isValid());                   // contains 0
        STATIC_CHECK_FALSE(Shape{0}.isValid());                         // contains 0
        STATIC_CHECK_FALSE(Shape{0, 0, 0}.isValid());                   // contains many 0s
        STATIC_CHECK_FALSE(Shape{-1, -1}.isValid());                    // contains multiple -1s
        STATIC_CHECK_FALSE(Shape{-1, 2, 3, -1, 4}.isValid());           // contains multiple -1s
        STATIC_CHECK_FALSE(Shape{-1, 0, 1, 23, 4}.isValid());           // contains -1 and 0
    }
}

TEST_CASE("Shape construction", "[Shape]"){
    SECTION("From args", "Shapes created with array initialization should equal the same crated by passing args"){
        Shape<0> shape0 = {};
        CHECK(shape0 == Shape{});

        Shape<1> shape1 = {1};
        CHECK(shape1 == Shape{1});

        Shape<2> shape2 = {1, 2};
        CHECK(shape2 == Shape{1, 2});

        Shape<3> shape3 = {1, 2, 3};
        CHECK(shape3 == Shape{1, 2, 3});

        Shape<4> shape4 = {1, 2, 3, 4};
        CHECK(shape4 == Shape{1, 2, 3, 4});
    }
}

TEMPLATE_TEST_CASE("Shape construction from range", "[Shape][template]",
                   (std::array<int, 4>), (std::array<size_t, 4>), std::vector<int>, std::vector<size_t>) {
    GIVEN("A range of 4 values"){
        TestType range = {1, 2, 3, 4};

        THEN("The range has size 4"){
            REQUIRE(std::ranges::size(range) == 4);
        }

        WHEN("A shape is created from that range"){
            Shape<4> shape = Shape<4>{from_range, range};

            THEN("The shape also has size 4"){
                CHECK(shape.rank() == 4);
                CHECK(std::ranges::size(shape) == 4);
            }

            THEN("The shape has the same values") {
                CHECK(shape[0] == 1);
                CHECK(shape[1] == 2);
                CHECK(shape[2] == 3);
                CHECK(shape[3] == 4);
            }
        }

        WHEN("A shape is created from a subset of the range"){
            THEN("It throws an error"){
                CHECK_THROWS(Shape<3>{from_range, range});
            }
        }
    }
}

SCENARIO("A Shape's dimensions can be indexed", "[Shape]"){
    GIVEN("A shape of rank 4") {
        Shape shape = Shape{1, 2, 3, 4};

        WHEN ("The dimensions are indexed") {
            THEN ("The dimension has the correct value") {
                CHECK(shape[0] == 1);
                CHECK(shape[1] == 2);
                CHECK(shape[2] == 3);
                CHECK(shape[3] == 4);
            }
        }

        WHEN ("An invalid, negative dimension is indexed") {
            THEN ("It throws") {
                CHECK_THROWS(shape[-1]);
            }
        }
        WHEN ("An invalid, too large dimension is indexed") {
            THEN ("It throws") {
                CHECK_THROWS(shape[42]);
            }
        }
    }
}

TEST_CASE("Shape n_elems", "[Shape][static]"){
    SECTION("Explicit shapes"){
        STATIC_CHECK(Shape{}.n_elems() == 1);
        STATIC_CHECK(Shape{5}.n_elems() == 5);
        STATIC_CHECK(Shape{5, 2}.n_elems() == 10);
        STATIC_CHECK(Shape{5, 1, 1, 2, 3}.n_elems() == 30);
    }

    SECTION("Implicit shapes"){
        STATIC_CHECK(Shape{-1}.n_elems() == 1);
        STATIC_CHECK(Shape{-1, 2}.n_elems() == 2);
        STATIC_CHECK(Shape{2, -1}.n_elems() == 2);
        STATIC_CHECK(Shape{23, 4, 5, -1, 7}.n_elems() == 3220);
    }
}

TEST_CASE("Shape deduction", "[Shape][static]"){
    SECTION("Valid, compatible shapes") {
        {
            Shape Explicit = Shape{10};
            Shape Implicit = Shape{-1};
            Shape Deduced = deduceShape(Explicit, Implicit);
            CHECK(Deduced == Shape{10});
        }
        {
            Shape Explicit = Shape{2, 3, 5};
            Shape Implicit = Shape{5, -1};
            Shape Deduced = deduceShape(Explicit, Implicit);
            CHECK(Deduced == Shape{5, 6});
        }
        {
            Shape Explicit = Shape{2, 2, 6, 7, 1};
            Shape Implicit = Shape{1, -1, 21};
            Shape Deduced = deduceShape(Explicit, Implicit);
            CHECK(Deduced == Shape{1, 8, 21});
        }
    }

    SECTION ("Incompatible shapes")
    {
        {
            Shape Explicit = Shape{10};
            Shape Implicit = Shape{3, -1};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{2, 3, 5};
            Shape Implicit = Shape{3, -1, 7};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{2};
            Shape Implicit = Shape{-1, 3, 5, 7};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
    }

    SECTION ("Using explicit shape as implicit")
    {
        {
            Shape Explicit = Shape{10};
            Shape Implicit = Shape{10};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{2, 3};
            Shape Implicit = Shape{10};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{10};
            Shape Implicit = Shape{2, 3};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{2, 5, 11};
            Shape Implicit = Shape{3, 7};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
    }

    SECTION ("Using implicit shape as explicit")
    {
        {
            Shape Explicit = Shape{3, -1};
            Shape Implicit = Shape{3, -1};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{7, -1, 3};
            Shape Implicit = Shape{3, -1};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{3, -1};
            Shape Implicit = Shape{7, -1, 3};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }{
            Shape Explicit = Shape{2, 5, -1, 11};
            Shape Implicit = Shape{3, 7, -1};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
    }
    SECTION ("Using invalid shapes")
    {
        {
            Shape Explicit = Shape{3, -1, -1};
            Shape Implicit = Shape{3, -1};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{7, 0, 3};
            Shape Implicit = Shape{3, -1};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{3, 1};
            Shape Implicit = Shape{3, -1, -1};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
        {
            Shape Explicit = Shape{3, 1};
            Shape Implicit = Shape{7, 0, 3};
            CHECK_THROWS(deduceShape(Explicit, Implicit));
        }
    }
}

TEST_CASE("Shape augment from args", "[Shape]"){
    {
        Shape before = Shape {1, 2, 3};
        Shape after = before.augmented(4);
        CHECK(after == Shape{1, 2, 3, 4});
    }
    {
        Shape before = Shape {1, 2, 3};
        Shape after = before.augmented(4, 5, 6, 7);
        CHECK(after == Shape{1, 2, 3, 4, 5, 6, 7});
    }
    {
        Shape before = Shape {};
        Shape after = before.augmented(1, 2, 3);
        CHECK(after == Shape{1, 2, 3});
    }
}

TEMPLATE_TEST_CASE("Shape augmented from range", "[Shape][template]",
                   (std::array<int, 4>), (std::array<size_t, 4>), std::vector<int>, std::vector<size_t>) {
    GIVEN("A range of 4 values and a shape of 3 values"){
        TestType range = {4, 5, 6, 7};
        Shape<3> smallShape = {1, 2, 3};

        THEN("The range has size 4 and the shape has size 3"){
            REQUIRE(std::ranges::size(range) == 4);
            REQUIRE(smallShape.rank() == 3);
        }

        WHEN("A shape is created from that range"){
            Shape<7> shape = smallShape.augmented<7>(from_range, range);

            THEN("The shape now has size 7"){
                CHECK(shape.rank() == 7);
                CHECK(std::ranges::size(shape) == 7);
            }
            THEN("The shape has the new values") {
                CHECK(shape[0] == 1);
                CHECK(shape[1] == 2);
                CHECK(shape[2] == 3);
                CHECK(shape[3] == 4);
                CHECK(shape[4] == 5);
                CHECK(shape[5] == 6);
                CHECK(shape[6] == 7);
            }
        }

        WHEN("A shape is created from a subset of that range"){
            THEN("It throws and error"){
                CHECK_THROWS(smallShape.augmented<5>(from_range, range));
            }
        }
    }
}

SCENARIO("Shapes can be demoted", "[Shape]"){
    GIVEN("A shape of rank 4") {
        Shape before = Shape{1, 2, 3, 4};

        WHEN ("The shape is demoted to rank 2") {
            Shape after = before.demoted<2>();
            THEN ("The shape has rank 2") {
                CHECK(after.rank() == 2);
            }
            THEN ("The shape has the same values") {
                CHECK(after[0] == 1);
                CHECK(after[1] == 2);
            }
        }

        WHEN ("The shape is demoted to rank 0") {
            Shape after = before.demoted<0>();
            THEN ("The shape has rank 0") {
                CHECK(after.rank() == 0);
            }
        }

        // This section shouldn't even compile
//            WHEN ("The shape is demotes to rank 5") {
//                CHECK_THROWS(before.demoted<5>());
//            }
    }
}

SCENARIO("A Shape's dimensions can be replaced", "[Shape]"){
    GIVEN("A shape of rank 4") {
        Shape before = Shape{1, 2, 3, 4};

        WHEN ("the first dimension is replaced") {
            Shape after = before.replace(0, 42);
            THEN ("The shape still has rank 4") {
                CHECK(after.rank() == 4);
            }
            THEN ("The shape has the same values") {
                CHECK(after[0] == 42);
                CHECK(after[1] == 2);
                CHECK(after[2] == 3);
                CHECK(after[3] == 4);
            }
        }

        WHEN ("the last dimension is replaced") {
            Shape after = before.replace(3, 42);
            THEN ("The shape still has rank 4") {
                CHECK(after.rank() == 4);
            }
            THEN ("The shape has the same values") {
                CHECK(after[0] == 1);
                CHECK(after[1] == 2);
                CHECK(after[2] == 3);
                CHECK(after[3] == 42);
            }
        }

        WHEN ("An invalid, negative dimension is replaced") {
            THEN ("It throws") {
                CHECK_THROWS(before.replace(-1, 42));
            }
        }
        WHEN ("An invalid, too large dimension is replaced") {
            THEN ("It throws") {
                CHECK_THROWS(before.replace(6, 42));
            }
        }
    }
}

