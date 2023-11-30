//
// Created by Amy Fetzner on 11/19/2023.
//

#include "Tensor_private.h"

using namespace TensorII::Core;

//region constructors
template<Scalar DType, typename Shape_, class Allocator>
Tensor<DType, Shape_, Allocator>::Tensor(TensorInitializer<DType, Shape_>::Array& array) {
    memcpy_s(data_, size_in_bytes(), array, size_in_bytes());
}

template<Scalar DType, typename Shape_, class Allocator>
Tensor<DType, Shape_, Allocator>::Tensor(TensorInitializer<DType, Shape_>&& initializer)
: Tensor<DType, Shape_, Allocator>::Tensor(initializer.values) {}

template<Scalar DType, typename Shape_, class Allocator>
constexpr DType* Tensor<DType, Shape_, Allocator>::data() noexcept { return &data_[0]; }

template<Scalar DType, typename Shape_, class Allocator>
constexpr const DType* Tensor<DType, Shape_, Allocator>::data() const noexcept { return &data_[0]; }
//endregion constructors


//region 0D tensor specialization
//region constructors
template<Scalar DType, class Allocator>
Tensor<DType, Shape<>, Allocator>::Tensor(DType value) : data_(value) {}

template<Scalar DType, class Allocator>
Tensor<DType, Shape<>, Allocator>::Tensor(TensorInitializer<DType, Shape>&& initializer)
: Tensor<DType, Shape, Allocator>::Tensor(initializer.value) {}
//endregion constructors

template<Scalar DType, class Allocator>
constexpr DType* Tensor<DType, Shape<>, Allocator>::data() noexcept { return &data_; }

template<Scalar DType, class Allocator>
constexpr const DType* Tensor<DType, Shape<>, Allocator>::data() const noexcept { return &data_; }

//endregion 0D specialization