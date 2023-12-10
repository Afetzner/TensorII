//
// Created by Amy Fetzner on 12/10/2023.
//

#include "TensorII/private/AnyShape.h"
#include "Catch2/catch_test_macros.hpp"

using namespace TensorII::Core;

TEST_CASE("AnyShape operator=", "[Shape]") {
    AnyShape<10> anyShape;
    // Unset
    CHECK_THROWS(anyShape.rank());
    CHECK_THROWS(anyShape.size());
    CHECK(!anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // Set a value
    anyShape = Shape{1, 2, 3};
    CHECK(anyShape.rank() == 3);
    CHECK(anyShape.size() == 6);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape == Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // Go up a rank
    anyShape = Shape{10, 20, 30, 40};
    CHECK(anyShape.rank() == 4);
    CHECK(anyShape.size() == 240'000);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{10, 20});
    CHECK(anyShape != Shape{10, 20, 30});
    CHECK(anyShape == Shape{10, 20, 30, 40});

    // Go down a rank
    anyShape = Shape{1, 2};
    CHECK(anyShape.rank() == 2);
    CHECK(anyShape.size() == 2);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape == Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // reset
    anyShape.reset();
    CHECK_THROWS(anyShape.rank());
    CHECK_THROWS(anyShape.size());
    CHECK(!anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // another value
    anyShape = Shape{1, 2, 3};
    CHECK(anyShape.rank() == 3);
    CHECK(anyShape.size() == 6);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape == Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});
}

TEST_CASE("AnyShape emplace values", "[Shape]") {
    AnyShape<10> anyShape;
    // Unset
    CHECK_THROWS(anyShape.rank());
    CHECK_THROWS(anyShape.size());
    CHECK(!anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // Set a value
    anyShape.emplace(1, 2, 3);
    CHECK(anyShape.rank() == 3);
    CHECK(anyShape.size() == 6);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape == Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // Go up a rank
    anyShape.emplace(10, 20, 30, 40);
    CHECK(anyShape.rank() == 4);
    CHECK(anyShape.size() == 240'000);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{10, 20});
    CHECK(anyShape != Shape{10, 20, 30});
    CHECK(anyShape == Shape{10, 20, 30, 40});

    // Go down a rank
    anyShape.emplace(1, 2);
    CHECK(anyShape.rank() == 2);
    CHECK(anyShape.size() == 2);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape == Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // reset
    anyShape.reset();
    CHECK_THROWS(anyShape.rank());
    CHECK_THROWS(anyShape.size());
    CHECK(!anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // another value
    anyShape.emplace(1, 2, 3);
    CHECK(anyShape.rank() == 3);
    CHECK(anyShape.size() == 6);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape == Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});
}

TEST_CASE("AnyShape emplace array", "[Shape]") {
    AnyShape<10> anyShape;
    // Unset
    CHECK_THROWS(anyShape.rank());
    CHECK_THROWS(anyShape.size());
    CHECK(!anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // Set a value
    anyShape.emplace({1, 2, 3});
    CHECK(anyShape.rank() == 3);
    CHECK(anyShape.size() == 6);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape == Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // Go up a rank
    anyShape.emplace({10, 20, 30, 40});
    CHECK(anyShape.rank() == 4);
    CHECK(anyShape.size() == 240'000);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{10, 20});
    CHECK(anyShape != Shape{10, 20, 30});
    CHECK(anyShape == Shape{10, 20, 30, 40});

    // Go down a rank
    anyShape.emplace({1, 2});
    CHECK(anyShape.rank() == 2);
    CHECK(anyShape.size() == 2);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape == Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // reset
    anyShape.reset();
    CHECK_THROWS(anyShape.rank());
    CHECK_THROWS(anyShape.size());
    CHECK(!anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape != Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});

    // another value
    anyShape.emplace({1, 2, 3});
    CHECK(anyShape.rank() == 3);
    CHECK(anyShape.size() == 6);
    CHECK(anyShape.isValidExplicit());
    CHECK(!anyShape.isValidImplicit());
    CHECK(anyShape != Shape{1, 2});
    CHECK(anyShape == Shape{1, 2, 3});
    CHECK(anyShape != Shape{1, 2, 3, 4});
}