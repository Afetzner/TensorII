//
// Created by Amy Fetzner on 11/19/2023.
//
#ifndef TENSOR_TENSOR_1_H
#define TENSOR_TENSOR_1_H

#include <concepts>
#include <memory>
#include <array>
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/TensorInitializer.h"
#include "TensorII/TensorIndex.h"

namespace TensorII::Core {

    template <Scalar DType>
    using TensorDefaultAllocator = std::allocator<DType>;

    template <Scalar DType, auto shape_, typename Allocator = TensorDefaultAllocator<DType>>
    class Tensor{
    public:
        consteval Shape<shape_.rank()> shape() { return shape_; };

        static constexpr tensorSize size() noexcept { return shape_.size(); };

        static constexpr tensorSize size_in_bytes() noexcept { return size() * sizeof(DType); };

        explicit Tensor(typename Private::TensorInitializer<DType, shape_>::Array&);

        explicit Tensor(Private::TensorInitializer<DType, shape_>&&);

        constexpr DType* data() noexcept;

        constexpr const DType* data() const noexcept;

    private:
        using Array = std::array<DType, size()>;
        struct ArrayDeleter {
            void operator()(std::array<DType, size()>* a);
        };
        std::unique_ptr<Array, ArrayDeleter> data_;
    };

    template <Scalar DType, Shape oldShape, Shape newShape>
            requires (oldShape.size() == newShape.size()
                      && oldShape.isValidExplicit()
                      && newShape.isValidExplicit())
    Tensor<DType, newShape>&
    reshape(Tensor<DType, oldShape>& t);

    template <Scalar DType, Shape oldShape, Shape newShape>
    requires (oldShape.size() == newShape.size()
              && oldShape.isValidExplicit()
              && newShape.isValidImplicit())
    Tensor<DType, deduceShape(oldShape, newShape)>&
    reshape(Tensor<DType, oldShape>& t);


    // 0D tensor
    template <Scalar DType, typename Allocator>
    class Tensor<DType, Shape<0>{}, Allocator> {
    public:
        consteval Shape<0> shape() { return Shape{}; };

        Tensor(DType value); // NOLINT(google-explicit-constructor)
        explicit Tensor(Private::TensorInitializer<DType, Shape<0>{}>&& initializer);

        Tensor(const Tensor&) = delete;

        Tensor(Tensor&&) = delete;

        static constexpr tensorSize size() noexcept { return Shape<0>::size(); };
        static constexpr tensorSize size_in_bytes() noexcept { return size() * sizeof(DType); };
        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

    private:
        DType data_;
    };

} // TensorII::Core

#endif //TENSOR_TENSOR_1_H
