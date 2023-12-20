//
// Created by Amy Fetzner on 11/19/2023.
//
#ifndef TENSOR_TENSOR_PRIVATE_H
#define TENSOR_TENSOR_PRIVATE_H

#include <concepts>
#include <memory>
#include <array>

#include "TensorII/Types.h"
#include "Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/TensorInitializer.h"
#include "TensorII/private/TensorIndex.h"

namespace TensorII::Core {

    template <Scalar DType, auto shape_, typename Allocator>
    class Tensor{
        Tensor();
    public:
        explicit Tensor(typename Private::TensorInitializer<DType, shape_>::Array&);
        explicit Tensor(Private::TensorInitializer<DType, shape_>&&);

        template <Util::SizedContainerCompatibleRange<DType> Range>
        explicit Tensor(from_range_t, Range&&);

        Tensor(const Tensor&) = delete;
        Tensor(Tensor&&) noexcept;

        Tensor& operator=(const Tensor&) = delete;
        Tensor& operator=(Tensor&&) noexcept;

        constexpr Shape<shape_.rank()> shape();

        static constexpr tensorSize size() noexcept;
        static constexpr tensorSize size_in_bytes() noexcept;

        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

    private:
        using Array = std::array<DType, size()>;
        struct ArrayDeleter {
            void operator()(Array* a);
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
        Tensor();
    public:
        Tensor(DType value); // NOLINT(google-explicit-constructor)
        explicit Tensor(Private::TensorInitializer<DType, Shape<0>{}>&& initializer);

        template <Util::SizedContainerCompatibleRange<DType> Range>
        explicit Tensor(from_range_t, Range&&);

        Tensor(const Tensor&);
        Tensor(Tensor&&) noexcept;

        Tensor& operator=(const Tensor&);
        Tensor& operator=(Tensor&&) noexcept;

        constexpr Shape<0> shape() noexcept;

        static constexpr tensorSize size() noexcept;
        static constexpr tensorSize size_in_bytes() noexcept;

        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

    private:
        DType data_;
    };

} // TensorII::Core

#endif //TENSOR_TENSOR_H
