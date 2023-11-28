//
// Created by Amy Fetzner on 11/19/2023.
//
#ifndef TENSOR_TENSOR_1_H
#define TENSOR_TENSOR_1_H

#include <concepts>
#include <memory>
#include <array>
#include "Shape.h"
#include "TensorDType.h"
#include "TensorInitializer.h"

namespace TensorII::Core {

    template <TensorDType DType, typename Shape_, class Allocator = std::allocator<DType>>
    class Tensor{
        std::array<DType, Shape_::size> data;

    public:
        explicit Tensor(TensorInitializer<DType, Shape_>::Array& array) {};

        explicit Tensor(TensorInitializer<DType, Shape_> initializer)
                : Tensor(initializer.values) {};

        Tensor(const Tensor&) = delete;
        Tensor(Tensor&&) = delete;
    };

    // 0D tensor
    template <TensorDType DType, class Allocator>
    class Tensor<DType, Shape<>, Allocator> {
        DType data;
    public:
        explicit Tensor(DType value) : data(value) {}
        explicit Tensor(TensorInitializer<DType, Shape<>> initializer)
                : Tensor(initializer.value) {};

        Tensor(const Tensor&) = delete;
        Tensor(Tensor&&) = delete;
    };

    //region toTensor
    // 0 dimensions
    template<TensorDType DType,
            class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<>, Allocator>
    toTensor(DType value)
    {
        return Tensor<DType, Shape<>, Allocator>
                (TensorInitializer<DType, Shape<>>(value));
    }

    // 1 dimension
    template<TensorDType DType,
            tensorDimension dimension,
            class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<dimension>, Allocator>
    toTensor(DType (&&array)[dimension])
    {
        return Tensor<DType, Shape<dimension>, Allocator>(TensorInitializer<DType, Shape<dimension>>(array));
    }

    // 2 dimensions
    template<TensorDType DType,
            tensorDimension d1,
            tensorDimension d2,
            class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<d1, d2>, Allocator>
    toTensor( DType (&&array)[d1][d2])
    {
        using Shape = Shape<d1, d2>;
        return Tensor<DType, Shape, Allocator> (TensorInitializer<DType, Shape>(array));
    }

    // 3 dimensions
    template<TensorDType DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3,
            class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<d1, d2, d3>, Allocator>
    toTensor( DType (&&array)[d1][d2][d3])
    {
        using Shape = Shape<d1, d2, d3>;
        return Tensor<DType, Shape, Allocator> (TensorInitializer<DType, Shape>(array));
    }

    // 4 dimensions
    template<TensorDType DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3,
            tensorDimension d4,
            class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<d1, d2, d3, d4>, Allocator>
    toTensor( DType (&&array)[d1][d2][d3][d4])
    {
        using Shape = Shape<d1, d2, d3, d4>;
        return Tensor<DType, Shape, Allocator> (TensorInitializer<DType, Shape>(array));
    }
    //endregion toTensor

    template<TensorDType DType, typename Shape_>
    Tensor(TensorInitializer<DType, Shape_>) -> Tensor<DType, Shape_>;

} // TensorII::Core

#endif //TENSOR_TENSOR_1_H
