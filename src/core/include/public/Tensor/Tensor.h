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

namespace TensorTech::Core {

    template <TensorDType DType, typename Shape_, class Allocator = std::allocator<DType>>
    class Tensor {
        std::array<DType, Shape_::size> data;
        explicit Tensor(typename TensorInitializer<DType, Shape_>::Values values) {}

    public:
        explicit Tensor(TensorInitializer<DType, Shape_>::Initializer initializer)
        requires (Shape_::ndim == 1)
            : Tensor(typename TensorInitializer<DType, Shape_>::Values(initializer)) {};

        explicit Tensor(TensorInitializer<DType, Shape_>::Initializer initializer)
        requires (Shape_::ndim > 1)
                : Tensor(typename TensorInitializer<DType, Shape_>::Values(
                        initializer, Shape_::size * sizeof(DType))) {};

        explicit Tensor(TensorInitializer<DType, Shape_> initializer)
                : Tensor(initializer.values) {};

        Tensor(const Tensor&) = delete;
        Tensor(Tensor&&) = delete;
    };

    // Specialization for 0D tensor
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
    template<TensorDType DType, class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<>, Allocator>
    toTensor(DType value)
    {
        return Tensor<DType, Shape<>, Allocator>
                (TensorInitializer<DType, Shape<>>(value));
    }

    // 1 dimension
    template<TensorDType DType, tensorShapeDim N, class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<N>, Allocator>
    toTensor(DType const (&list)[N])
    {
        return Tensor<DType, Shape<N>, Allocator>
                (TensorInitializer<DType, Shape<N>>(list));
    }

    // >1 dimensions
    template<TensorDType DType, tensorShapeDim N, tensorShapeDim ... rest, class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<N, rest...>, Allocator>
    toTensor(TensorInitializer<DType, Shape<rest...>> const (&list)[N] )
    {
        return Tensor<DType, Shape<N, rest...>, Allocator>
                (TensorInitializer<DType, Shape<N, rest...>>(list));
    }
    //endregion toTensor

    template<TensorDType DType, typename Shape_>
    Tensor(TensorInitializer<DType, Shape_>) -> Tensor<DType, Shape_>;

} // TensorTech::Core

#endif //TENSOR_TENSOR_1_H
