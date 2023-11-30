//
// Created by Amy Fetzner on 11/19/2023.
//
#ifndef TENSOR_TENSOR_1_H
#define TENSOR_TENSOR_1_H

#include <concepts>
#include <memory>
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/TensorInitializer.h"

namespace TensorII::Core {

    template <Scalar DType, typename Shape_, class Allocator = std::allocator<DType>>
    class Tensor{
    public:
        using Shape = Shape_;

        static constexpr tensorSize size() noexcept { return Shape::size; };
        static constexpr tensorSize size_in_bytes() noexcept { return size() * sizeof(DType); };

        explicit Tensor(TensorInitializer<DType, Shape>::Array& array);
        explicit Tensor(TensorInitializer<DType, Shape>&& initializer);

        template<typename NewShape, class NewAllocator = Allocator>
        operator Tensor<NewShape, NewAllocator>();
        Tensor(Tensor&&) = delete;

        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

    private:
        DType data_[size()];
    };

    // 0D tensor
    template <Scalar DType, class Allocator>
    class Tensor<DType, Shape<>, Allocator> {
    public:
        using Shape = Shape<>;

        Tensor(DType value); // NOLINT(google-explicit-constructor)
        explicit Tensor(TensorInitializer<DType, Shape>&& initializer);
        Tensor(const Tensor&) = delete;
        Tensor(Tensor&&) = delete;

        static constexpr tensorSize size() noexcept { return Shape::size; };
        static constexpr tensorSize size_in_bytes() noexcept { return size() * sizeof(DType); };
        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

    private:
        DType data_;
    };

    //region toTensor
    // 0 dimensions
    template<Scalar DType,
            class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<>, Allocator>
    toTensor(DType value)
    {
        return Tensor<DType, Shape<>, Allocator>(TensorInitializer<DType, Shape<>>(value));
    }

    // 1 dimension
    template<Scalar DType,
            tensorDimension dimension,
            class Allocator = std::allocator<DType>>
    Tensor<DType, Shape<dimension>, Allocator>
    toTensor(DType (&&array)[dimension])
    {
        return Tensor<DType, Shape<dimension>, Allocator>(TensorInitializer<DType, Shape<dimension>>(array));
    }

    // 2 dimensions
    template<Scalar DType,
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
    template<Scalar DType,
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
    template<Scalar DType,
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

} // TensorII::Core

#endif //TENSOR_TENSOR_1_H
