//
// Created by Amy Fetzner on 12/21/2023.
//

#include "TensorII/private/TensorInitializer.h"
#include "Catch2/catch_test_macros.hpp"

using namespace TensorII::Core;
using namespace TensorII::Core::Private;

SCENARIO("A 0D TensorInitializer can be created and iterated", "[TensorInitializer]"){
    WHEN("An r-value is used"){
        TensorInitializer<int, Shape{}> tensorInit {1};
        THEN("It has the correct values"){
            CHECK(*tensorInit.begin() == 1);
            CHECK(++tensorInit.begin() == tensorInit.end());
        }
    }
    WHEN("An l-value is used"){
        int x = 1;
        TensorInitializer<int, Shape{}> tensorInit {x};
        THEN("It has the correct values"){
            CHECK(*tensorInit.begin() == 1);
            CHECK(++tensorInit.begin() == tensorInit.end());
        }
    }
    WHEN("An l-value is used in a constant expression"){
        static constexpr int x = 1;
        constexpr TensorInitializer<int, Shape{}> tensorInit {x};
        THEN("It has the correct values"){
            STATIC_CHECK(*tensorInit.begin() == 1);
            STATIC_CHECK(++tensorInit.begin() == tensorInit.end());
        }
    }
}

SCENARIO("A 1D TensorInitializer can be created and iterated", "[TensorInitializer]"){
    static constexpr int flat[4] = {1, 2, 3, 4};
    WHEN("An r-value array is used"){
        TensorInitializer<int, Shape{4}> tensorInit ({1, 2, 3, 4});
        THEN("It has the correct values"){
            CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
    WHEN("An l-value array is used"){
        const int arr[4] = {1, 2, 3, 4};
        TensorInitializer<int, Shape{4}> tensorInit {arr};
        THEN("It has the correct values"){
            CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
    // Cannot pass an r-value array in a constant expression since the array would be a temporary
    WHEN("An l-value array is used in a constant expression"){
        static constexpr int arr[4] = {1, 2, 3, 4};  // Must be static constexpr array to pass it to constexpr constructor
        constexpr TensorInitializer<int, Shape{4}> tensorInit {arr};
        THEN("It has the correct values"){
            STATIC_CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
}

SCENARIO("A 2D TensorInitializer can be created and iterated", "[TensorInitializer]"){
    static constexpr int flat[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    WHEN("An r-value array is used"){
        TensorInitializer<int, Shape{2, 4}> tensorInit ({{1, 2, 3, 4},
                                                         {5, 6, 7, 8}});
        THEN("It has the correct values"){
            CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
    WHEN("An l-value array is used"){
        const int arr[2][4] = {{1, 2, 3, 4},
                               {5, 6, 7, 8}};
        TensorInitializer<int, Shape{2, 4}> tensorInit {arr};
        THEN("It has the correct values"){
            CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
    // Cannot pass an r-value array in a constant expression since the array would be a temporary
    WHEN("An l-value array is used in a constant expression"){
        static constexpr int arr[2][4] = {{1, 2, 3, 4},
                                          {5, 6, 7, 8}};  // Must be static constexpr array to pass it to constexpr constructor
        constexpr TensorInitializer<int, Shape{2, 4}> tensorInit {arr};
        THEN("It has the correct values"){
            STATIC_CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
}

SCENARIO("A 3D TensorInitializer can be created and iterated", "[TensorInitializer]"){
    static constexpr int flat[24] = {1,  2,  3,  4,  5,  6,  7,  8,
                                     9,  10, 11, 12, 13, 14, 15, 16,
                                     17, 18, 19, 20, 21, 22, 23, 24};
    WHEN("An r-value array is used"){
        TensorInitializer<int, Shape{3, 2, 4}> tensorInit ({{{1,  2,  3,  4},  {5,  6,  7,  8}},
                                                            {{9,  10, 11, 12}, {13, 14, 15, 16}},
                                                            {{17, 18, 19, 20}, {21, 22, 23, 24}}});
        THEN("It has the correct values"){
            CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
    WHEN("An l-value array is used"){
        const int arr[3][2][4] = {{{1,  2,  3,  4},  {5,  6,  7,  8}},
                                  {{9,  10, 11, 12}, {13, 14, 15, 16}},
                                  {{17, 18, 19, 20}, {21, 22, 23, 24}}};
        TensorInitializer<int, Shape{3, 2, 4}> tensorInit {arr};
        THEN("It has the correct values"){
            CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
        // Cannot pass an r-value array in a constant expression since the array would be a temporary
    WHEN("An l-value array is used in a constant expression"){
        static constexpr int arr[3][2][4] = {{{1,  2,  3,  4},  {5,  6,  7,  8}},
                                             {{9,  10, 11, 12}, {13, 14, 15, 16}},
                                             {{17, 18, 19, 20}, {21, 22, 23, 24}}};
        constexpr TensorInitializer<int, Shape{3, 2, 4}> tensorInit {arr};
        THEN("It has the correct values"){
            STATIC_CHECK(std::ranges::equal(flat, tensorInit));
        }
    }
}
