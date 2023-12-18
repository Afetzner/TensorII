//
// Created by Amy Fetzner on 11/19/2023.
//

#ifndef TENSOR_TENSOR_TPP
#define TENSOR_TENSOR_TPP

#include "TensorII/private/headers/Tensor_private.h"

namespace TensorII::Core {

    template<Scalar DType, auto shape_, typename Allocator>
    Tensor<DType, shape_, Allocator>::Tensor() {
        data_ = std::make_unique<Array, ArrayDeleter>();
    }

    template<Scalar DType, auto shape_, typename Allocator>
    Tensor<DType, shape_, Allocator>::Tensor(typename Private::TensorInitializer<DType, shape_>::Array &array) {
        static_assert(sizeof(Array) == sizeof(DType) * size(),
                      "Mismatch of sizeof(DType[N]) and std::array<DType, N>, cursed compiler?");
        Allocator allocator;
        Array *array_p = new(allocator.allocate(size())) Array;
        data_ = std::unique_ptr<Array, ArrayDeleter>(array_p);
        memcpy_s(data_.get(), size_in_bytes(), array, size_in_bytes());
    }

    template<Scalar DType, auto shape_, typename Allocator>
    Tensor<DType, shape_, Allocator>::Tensor(Private::TensorInitializer<DType, shape_> &&initializer)
    : Tensor<DType, shape_>::Tensor(initializer.values)
    {}

    template<Scalar DType, auto shape_, typename Allocator>
    template<Util::ContainerCompatibleRange<DType> Range>
    Tensor<DType, shape_, Allocator>::Tensor(from_range_t, Range&& range) {
        std::ranges::copy_n(range.being(), size(), data_->begin());
    }

    template<Scalar DType, auto shape_, typename Allocator>
    Tensor<DType, shape_, Allocator>::Tensor(Tensor && other) noexcept
    : data_(std::exchange(other.cstring, nullptr))
    {}

    template<Scalar DType, auto shape_, typename Allocator>
    Tensor<DType, shape_, Allocator>& Tensor<DType, shape_, Allocator>::operator=(Tensor && other) noexcept {
        std::swap(data_, other.data_);
        return *this;
    }

    template<Scalar DType, auto shape_, typename Allocator>
    constexpr Shape<shape_.rank()> Tensor<DType, shape_, Allocator>::shape() {
        return shape_;
    }

    template<Scalar DType, auto shape_, typename Allocator>
    constexpr tensorSize Tensor<DType, shape_, Allocator>::size() noexcept {
        return shape_.size();
    }

    template<Scalar DType, auto shape_, typename Allocator>
    constexpr tensorSize Tensor<DType, shape_, Allocator>::size_in_bytes() noexcept {
        return size() * sizeof(DType);
    }


    template<Scalar DType, auto shape, typename Allocator>
    void Tensor<DType, shape, Allocator>::ArrayDeleter::operator()(std::array<DType, size()> *a) {
        Allocator allocator;
        allocator.deallocate((DType *) a, size());
    }

    template<Scalar DType, auto shape_, typename Allocator>
    constexpr DType *Tensor<DType, shape_, Allocator>::data() noexcept {
        return data_->data();
    }

    template<Scalar DType, auto shape_, typename Allocator>
    constexpr const DType *Tensor<DType, shape_, Allocator>::data() const noexcept {
        return data_->data();
    }

    template<auto newShape, auto oldShape, Scalar DType>
    requires (oldShape.size() == newShape.size()
              && oldShape.isValidExplicit()
              && newShape.isValidExplicit())
    Tensor<DType, newShape> &
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }

    template<Shape newShape, Shape oldShape, Scalar DType>
    requires (oldShape.size() == newShape.size()
              && oldShape.isValidExplicit()
              && newShape.isValidImplicit())
    Tensor<DType, deduceShape(oldShape, newShape)>&
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }


    //region 0D tensor specialization
    template<Scalar DType, typename Allocator>
    Tensor<DType, Shape<0>{}, Allocator>::Tensor(DType value)
    : data_(value)
    {}

    template<Scalar DType, typename Allocator>
    Tensor<DType, Shape<0>{}, Allocator>::Tensor(Private::TensorInitializer<DType, Shape<0>{}, 0> &&initializer)
    : Tensor<DType, Shape<0>{}, Allocator>::Tensor(initializer.value)
    {}

    template<Scalar DType, typename Allocator>
    constexpr Shape<0> Tensor<DType, Shape<0>{}, Allocator>::shape() noexcept {
        return Shape<0>{};
    }

    template<Scalar DType, typename Allocator>
    constexpr tensorSize Tensor<DType, Shape<0>{}, Allocator>::size() noexcept {
        return Shape<0>{}.size();
    }

    template<Scalar DType, typename Allocator>
    constexpr tensorSize Tensor<DType, Shape<0>{}, Allocator>::size_in_bytes() noexcept {
        return size() * sizeof(DType);
    }

    template<Scalar DType, typename Allocator>
    constexpr DType *Tensor<DType, Shape<0>{}, Allocator>::data() noexcept {
        return &data_;
    }

    template<Scalar DType, typename Allocator>
    constexpr const DType *Tensor<DType, Shape<0>{}, Allocator>::data() const noexcept {
        return &data_;
    }
}

#endif //TENSOR_TENSOR_TPP
