//
// Created by Amy Fetzner on 11/19/2023.
//

#ifndef TENSOR_TENSOR_TPP
#define TENSOR_TENSOR_TPP

#include "TensorII/Tensor.h"

namespace TensorII::Core {

    template<Scalar DType, auto shape_>
    constexpr Tensor<DType, shape_>::Tensor()
    : data_ {}
    {}

    template<Scalar DType, auto shape_>
    constexpr Tensor<DType, shape_>::Tensor(typename Private::TensorInitializer<DType, shape_>::Array &array) {
        static_assert(sizeof(Array) == sizeof(decltype(array))); // assert c-array same size as std::array
        if (!std::is_constant_evaluated()){
            memmove_s(data_.data(), size_in_bytes(), array, size_in_bytes());
            return;
        }
    }

    template<Scalar DType, auto shape_>
    constexpr Tensor<DType, shape_>::Tensor(Private::TensorInitializer<DType, shape_> &&initializer)
            : Tensor<DType, shape_>::Tensor(from_range, initializer)
    {}

    template<Scalar DType, auto shape_>
    template<Util::SizedContainerCompatibleRange<DType> Range>
    constexpr Tensor<DType, shape_>::Tensor(from_range_t, Range && range) {
        std::ranges::copy_n(range.begin(), size(), data_.begin());
    }

    template<Scalar DType, auto shape_>
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
    constexpr Shape<shape_.rank()> Tensor<DType, shape_>::shape() {
        return shape_;
    }

    template<Scalar DType, auto shape_>
    constexpr tensorSize Tensor<DType, shape_>::size() noexcept {
        return shape_.n_elems();
    }

    template<Scalar DType, auto shape_>
    constexpr tensorSize Tensor<DType, shape_>::size_in_bytes() noexcept {
        return size() * sizeof(DType);
    }

    template<Scalar DType, auto shape_>
    constexpr DType *Tensor<DType, shape_>::data() noexcept {
        return data_.data();
    }

    template<Scalar DType, auto shape_>
    constexpr const DType *Tensor<DType, shape_>::data() const noexcept {
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


//    //region 0D tensor specialization
//    template<Scalar DType>
//    constexpr Tensor<DType, Shape<0>{}>::Tensor(DType value)
//    : data_(value)
//    {}
//
//    template<Scalar DType>
//    constexpr Tensor<DType, Shape<0>{}>::Tensor(Private::TensorInitializer<DType, Shape<0>{}, 0> &&initializer)
//    : Tensor<DType, Shape<0>{}>::Tensor(initializer.value)
//    {}
//
//
////    template<Scalar DType>
////    template<Util::SizedContainerCompatibleRange<DType> Range>
////    constexpr Tensor<DType, Shape<0>{}>::Tensor(from_range_t, Range&& range) {
////        data_ = *range.begin();
////    }
//
//    template<Scalar DType>
//    constexpr Shape<0> Tensor<DType, Shape<0>{}>::shape() noexcept {
//        return Shape<0>{};
//    }
//
//    template<Scalar DType>
//    constexpr tensorSize Tensor<DType, Shape<0>{}>::size() noexcept {
//        return Shape<0>{}.size();
//    }
//
//    template<Scalar DType>
//    constexpr tensorSize Tensor<DType, Shape<0>{}>::size_in_bytes() noexcept {
//        return size() * sizeof(DType);
//    }
//
//    template<Scalar DType>
//    constexpr DType *Tensor<DType, Shape<0>{}>::data() noexcept {
//        return &data_;
//    }
//
//    template<Scalar DType>
//    constexpr const DType *Tensor<DType, Shape<0>{}>::data() const noexcept {
//        return &data_;
//    }
}

#endif //TENSOR_TENSOR_TPP
