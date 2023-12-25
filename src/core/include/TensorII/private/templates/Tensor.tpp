//
// Created by Amy Fetzner on 11/19/2023.
//

#ifndef TENSOR_TENSOR_TPP
#define TENSOR_TENSOR_TPP

#include "TensorII/Tensor.h"

namespace TensorII::Core {

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor()
    : data_ {}
    {}

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor(typename Private::TensorInitializer<DType, shape_>::Array &array) {
        static_assert(sizeof(Array) == sizeof(decltype(array))); // assert c-array same size as std::array
        if (!std::is_constant_evaluated()){
            memmove_s(data_.data(), size_in_bytes(), array, size_in_bytes());
            return;
        }
    }

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor(Private::TensorInitializer<DType, shape_> &&initializer)
            : Tensor<DType, shape_>::Tensor(from_range, initializer)
    {}

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    template<Util::ContainerCompatibleRange<DType> Range>
    constexpr Tensor<DType, shape_>::Tensor(from_range_t, Range && range) {
        std::ranges::copy_n(range.begin(), size(), data_.begin());
    }

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor(Tensor && other) noexcept
    : data_{}
    {
        if (!std::is_constant_evaluated()){
            memmove_s(data_.data(), size_in_bytes(), other.data(), other.size_in_bytes());
            return;
        }
        for(size_t i = 0; i < Array::size(); i++){
            data_[i] = other.data_[i];
        }
    }

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>& Tensor<DType, shape_>::operator=(Tensor && other) noexcept {
        if (!std::is_constant_evaluated()){
            memmove_s(data_.data(), size_in_bytes(), other.data(), other.size_in_bytes());
            return *this;
        }
        for(size_t i = 0; i < Array::size(); i++){
            data_[i] = other.data_[i];
        }
        return *this;
    }

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::size_type Tensor<DType, shape_>::size() noexcept {
        return shape_.n_elems();
    }

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::size_type Tensor<DType, shape_>::size_in_bytes() noexcept {
        return size() * sizeof(DType);
    }

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::pointer Tensor<DType, shape_>::data() noexcept {
        return data_.data();
    }

    template<Scalar DType, auto shape_>
    requires (is_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::const_pointer Tensor<DType, shape_>::data() const noexcept {
        return data_.data();
    }

    template<auto newShape, auto oldShape, Scalar DType>
    requires (oldShape.n_elems() == newShape.n_elems()
              && oldShape.isValidExplicit()
              && newShape.isValidExplicit())
    Tensor<DType, newShape> &
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }

    template<Shape newShape, Shape oldShape, Scalar DType>
    requires (oldShape.n_elems() == newShape.n_elems()
              && oldShape.isValidExplicit()
              && newShape.isValidImplicit())
    Tensor<DType, deduceShape(oldShape, newShape)>&
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }
}

#endif //TENSOR_TENSOR_TPP
