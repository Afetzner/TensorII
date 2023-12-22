//
// Created by Amy Fetzner on 11/19/2023.
//
#ifndef TENSOR_TENSOR_PRIVATE_H
#define TENSOR_TENSOR_PRIVATE_H

#include <concepts>
#include <memory>
#include <array>

#include "TensorII/Types.h"
#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/TensorInitializer.h"
#include "TensorII/private/TensorIndex.h"

namespace TensorII::Core {

    template <Scalar DType, auto shape_>
    class Tensor{
    public:
        constexpr Tensor();
        explicit constexpr Tensor(typename Private::TensorInitializer<DType, shape_>::Array&);
        explicit constexpr Tensor(Private::TensorInitializer<DType, shape_>&&);

        template <Util::SizedContainerCompatibleRange<DType> Range>
        constexpr explicit Tensor(from_range_t, Range&&);

        // Copy not allowed
        constexpr Tensor(const Tensor&) = delete;
        constexpr Tensor& operator=(const Tensor&) = delete;

        constexpr Tensor(Tensor&&) noexcept;
        constexpr Tensor& operator=(Tensor&&) noexcept;

        constexpr Shape<shape_.rank()> shape();

        static constexpr tensorSize size() noexcept;
        static constexpr tensorSize size_in_bytes() noexcept;

        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

    private:
        using Array = std::array<DType, size()>;
        Array data_;
    };

    template <Scalar DType, Shape oldShape, Shape newShape>
    requires (oldShape.n_elems() == newShape.n_elems()
              && oldShape.isValidExplicit()
              && newShape.isValidExplicit())
    constexpr Tensor<DType, newShape>&
    reshape(Tensor<DType, oldShape>& t);

    template <Scalar DType, Shape oldShape, Shape newShape>
    requires (oldShape.n_elems() == newShape.n_elems()
              && oldShape.isValidExplicit()
              && newShape.isValidImplicit())
    constexpr Tensor<DType, deduceShape(oldShape, newShape)>&
    reshape(Tensor<DType, oldShape>& t);


//    // 0D tensor
//    template <Scalar DType>
//    class Tensor<DType, Shape<0>{}> {
//        constexpr Tensor();
//    public:
//        constexpr Tensor(DType value); // NOLINT(google-explicit-constructor)
//        explicit constexpr Tensor(Private::TensorInitializer<DType, Shape<0>{}>&& initializer);
//
//        template <Util::SizedContainerCompatibleRange<DType> Range>
//        explicit constexpr Tensor(from_range_t, Range&&);
//
//        constexpr Tensor(const Tensor&);
//        constexpr Tensor(Tensor&&) noexcept;
//
//        constexpr Tensor& operator=(const Tensor&);
//        constexpr Tensor& operator=(Tensor&&) noexcept;
//
//        constexpr Shape<0> shape() noexcept;
//
//        static constexpr tensorSize size() noexcept;
//        static constexpr tensorSize size_in_bytes() noexcept;
//
//        constexpr DType* data() noexcept;
//        constexpr const DType* data() const noexcept;
//
//    private:
//        DType data_;
//    };

} // TensorII::Core

#endif //TENSOR_TENSOR_H
