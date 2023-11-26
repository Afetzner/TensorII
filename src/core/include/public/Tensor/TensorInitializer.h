//
// Created by Amy Fetzner on 11/24/2023.
//
#ifndef TENSOR_TENSORINITIALIZER_H
#define TENSOR_TENSORINITIALIZER_H

#include <concepts>
#include <type_traits>
#include <span>
#include "Shape.h"
#include "TensorDType.h"

namespace TensorTech::Core{

    template<TensorDType DType, typename Shape_>
    struct TensorInitializer;

    //region TensorInitializer Definitions
    // 0 dimensions
    template<TensorDType DType, typename Shape_> requires (Shape_::ndim == 0)
    struct TensorInitializer<DType, Shape_> {
        using Value = DType;
        using Initializer = DType;
        Value value;

        TensorInitializer(Value value) // NOLINT(google-explicit-constructor)
        : value(value) {};
    };



    // 1 dimension
    template<TensorDType DType, typename Shape_> requires (Shape_::ndim == 1)
    struct TensorInitializer<DType, Shape_> {
        using ElemType = DType;
        using Values = std::span<const ElemType, Shape_::first_dim>;
        using Array = ElemType const [Shape_::first_dim];
        using Initializer = Array&;
        Values values;

        TensorInitializer(Values values) // NOLINT(google-explicit-constructor)
        : values(values) {}

        TensorInitializer(Initializer list) // NOLINT(google-explicit-constructor)
        : TensorInitializer(Values(list)) {}
    };

    // >1 dimension
    template<TensorDType DType, typename Shape_> requires (Shape_::ndim > 1)
    struct TensorInitializer<DType, Shape_> {
        using ShapeRemaining = typename Shape_::ShapeRemaining;
        using Values = std::span<const DType, Shape_::size>;
        using Array = TensorInitializer<DType, ShapeRemaining>::Array const [Shape_::first_dim];
        using Initializer = Array&;
        Values values;

        TensorInitializer(Values values) // NOLINT(google-explicit-constructor)
        : values(values) {};

        TensorInitializer(Initializer initializer) // NOLINT(google-explicit-constructor)
        : TensorInitializer(Values((int*)initializer, Shape_::size * sizeof(DType))) {}
    };

    //endregion TensorInitializer Definitions

    //region Template Deduction Guides
    // 0 Dimensions
//    template<TensorDType DType>
//    TensorInitializer(DType value)
//    -> TensorInitializer<DType, Shape<>>;

    // 1 Dimension
//    template<TensorDType DType, tensorShapeDim dimension>
//    TensorInitializer(DType const (&)[dimension])
//    -> TensorInitializer<DType, Shape<dimension>>;

    // >1 Dimension
//    template<TensorDType DType, tensorShapeDim first, tensorShapeDim... rest>
//    TensorInitializer(std::span<TensorInitializer<DType, Shape <rest...>>, first> span)
//    -> TensorInitializer<DType, Shape<span.size(), rest...>>;

    //endregion Template Deduction Guides
}

#endif //TENSOR_TENSORINITIALIZER_H
