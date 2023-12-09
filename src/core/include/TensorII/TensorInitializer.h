//
// Created by Amy Fetzner on 11/24/2023.
//
#ifndef TENSOR_TENSORINITIALIZER_H
#define TENSOR_TENSORINITIALIZER_H

#include <concepts>
#include <type_traits>
#include "Shape.h"
#include "TensorDType.h"

namespace TensorII::Core::Private {

    template<Scalar, Shape, tensorRank index = 0>
    struct TensorInitializer;

    //region TensorInitializer Definitions
    // 0 dimensions
    template<Scalar DType>
    struct TensorInitializer<DType, Shape<0>{}, 0> {
        DType value;
        TensorInitializer(DType value) // NOLINT(google-explicit-constructor)
        : value(value) {};
    };

    // >1 dimension
    template<Scalar DType, Shape shape, tensorRank index>
    requires (shape.rank() > 0 && index < shape.rank() - 1)
    struct TensorInitializer<DType, shape, index> {
    private:
        using LowerArray = typename TensorInitializer<DType, shape, index + 1>::Array;
    public:
        using Array = LowerArray const [shape[index]];
        Array& values;

        TensorInitializer(Array& values) // NOLINT(google-explicit-constructor)
        : values(values) {};
    };

    template<Scalar DType, Shape shape, tensorRank index>
    requires (shape.rank() > 0 && index == shape.rank() - 1)
    struct TensorInitializer<DType, shape, index> {
    public:
        using Array = DType const [shape[index]];
        Array& values;

        TensorInitializer(Array& values) // NOLINT(google-explicit-constructor)
                : values(values) {};
    };
    //endregion TensorInitializer Definitions
}

#endif //TENSOR_TENSORINITIALIZER_H
