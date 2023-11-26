//
// Created by Amy Fetzner on 11/24/2023.
//

#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

namespace TensorTech::Core {

    using tensorShapeDim = unsigned int;
    using tensorNDim = unsigned int;
    using tensorSize = unsigned int;

    template <tensorShapeDim ... Dimensions>
    struct Shape;

    // 0 dimensions
    template<>
    struct Shape<> {
        static constexpr tensorNDim ndim = 0;
        static constexpr tensorSize size = 1;
    };

    // 1 dimension
    template <tensorShapeDim first>
    struct Shape<first> {
        static constexpr tensorNDim ndim = 1;
        static constexpr tensorSize size = first;
        static constexpr tensorShapeDim first_dim = first;
        using ShapeRemaining = Shape<>;
    };

    // >1 dimensions
    template <tensorShapeDim first, tensorShapeDim ... rest>
    struct Shape<first, rest...> {
        static constexpr tensorNDim ndim = sizeof...(rest) + 1;
        static constexpr tensorSize size = (first * ... * rest);
        static constexpr tensorShapeDim first_dim = first;
        using ShapeRemaining = Shape<rest...>;
    };
}

#endif //TENSOR_SHAPE_H
