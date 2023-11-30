//
// Created by Amy Fetzner on 11/19/2023.
//

#include "Tensor_private.h"

using namespace TensorII::Core;

//region constructors
template<TensorDType DType, typename Shape_, class Allocator>
Tensor<DType, Shape_, Allocator>::Tensor(TensorInitializer<DType, Shape_>::Array& array) {
    memcpy_s(data_, size_in_bytes(), array, size_in_bytes());
}

template<TensorDType DType, typename Shape_, class Allocator>
Tensor<DType, Shape_, Allocator>::Tensor(TensorInitializer<DType, Shape_>&& initializer)
: Tensor<DType, Shape_, Allocator>::Tensor(initializer.values) {}

template<TensorDType DType, typename Shape_, class Allocator>
constexpr DType* Tensor<DType, Shape_, Allocator>::data() noexcept { return &data_[0]; }

template<TensorDType DType, typename Shape_, class Allocator>
constexpr const DType* Tensor<DType, Shape_, Allocator>::data() const noexcept { return &data_[0]; }
//endregion constructors


//region 0D tensor specialization
//region constructors
template<TensorDType DType, class Allocator>
Tensor<DType, Shape<>, Allocator>::Tensor(DType value) : data_(value) {}

template<TensorDType DType, class Allocator>
Tensor<DType, Shape<>, Allocator>::Tensor(TensorInitializer<DType, Shape>&& initializer)
: Tensor<DType, Shape, Allocator>::Tensor(initializer.value) {}
//endregion constructors

template<TensorDType DType, class Allocator>
constexpr DType* Tensor<DType, Shape<>, Allocator>::data() noexcept { return &data_; }

template<TensorDType DType, class Allocator>
constexpr const DType* Tensor<DType, Shape<>, Allocator>::data() const noexcept { return &data_; }

//endregion 0D specialization