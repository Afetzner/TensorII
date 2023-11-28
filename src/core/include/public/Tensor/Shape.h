//
// Created by Amy Fetzner on 11/24/2023.
//

#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

namespace TensorII::Core {

    using tensorDimension = size_t;
    using tensorRank = size_t;
    using tensorSize = size_t;

    template <tensorDimension ... Dimensions>
    struct Shape;

    // 0 dimensions
    template<>
    struct Shape<> {
        static constexpr tensorRank rank = 0;
        static constexpr tensorSize size = 1;
    };

    // 1 dimension
    template <tensorDimension first>
    struct Shape<first> {
        static constexpr tensorRank rank = 1;
        static constexpr tensorSize size = first;
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<>;
    };

    // >1 dimensions
    template <tensorDimension first, tensorDimension ... rest>
    struct Shape<first, rest...> {
        static constexpr tensorRank rank = sizeof...(rest) + 1;
        static constexpr tensorSize size = (first * ... * rest);
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<rest...>;
    };
}

#endif //TENSOR_SHAPE_H
