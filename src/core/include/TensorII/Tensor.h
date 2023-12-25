//
// Created by Amy Fetzner on 11/19/2023.
//
#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <concepts>
#include <memory>
#include <array>

#include "TensorII/Types.h"
#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/private/TensorInitializer.h"
#include "TensorII/private/TensorIndex.h"
#include "TensorII/private/ConceptUtil.h"

namespace TensorII::Core {

    template <Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    class Tensor {
    public:
        using value_type = DType;
        using size_type = tensorSize;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        constexpr Tensor();
        explicit constexpr Tensor(typename Private::TensorInitializer<DType, shape_>::Array&);
        explicit constexpr Tensor(Private::TensorInitializer<DType, shape_>&&);

        template <Util::ContainerCompatibleRange<DType> Range>
        constexpr explicit Tensor(from_range_t, Range&&);

        // Copy not allowed
        constexpr Tensor(const Tensor&) = delete;
        constexpr Tensor& operator=(const Tensor&) = delete;

        constexpr Tensor(Tensor&&) noexcept;
        constexpr Tensor& operator=(Tensor&&) noexcept;

        static constexpr tensorRank rank() noexcept { return shape_.rank(); }
        static constexpr Shape<shape_.rank()> shape() noexcept { return shape_; }

        static constexpr size_type size() noexcept;
        static constexpr size_type size_in_bytes() noexcept;

        constexpr pointer data() noexcept;
        constexpr const_pointer data() const noexcept;

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
} // TensorII::Core

#endif //TENSOR_TENSOR_H

#include "TensorII/private/templates/toTensor.tpp"
#include "TensorII/private/templates/Tensor.tpp" // NOLINT(bugprone-suspicious-include)
