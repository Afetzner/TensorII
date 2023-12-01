//
// Created by Amy Fetzner on 11/19/2023.
//

#include "Tensor_private.h"

using namespace TensorII::Core;

//region constructors
template<Scalar DType, typename Shape_, typename Allocator>
Tensor<DType, Shape_, Allocator>::Tensor(TensorInitializer<DType, Shape_>::Array& array) {
    static_assert(sizeof(Array) == sizeof(DType) * size(),
            "Mismatch of sizeof(DType[N]) and std::array<DType, N>, cursed compiler?");
    Allocator allocator;
    Array* array_p = new(allocator.allocate(size())) Array;
    data_ = std::unique_ptr<Array, TensorArrayDeleter<DType, size()>>(array_p);
    memcpy_s(data_.get(), size_in_bytes(), array, size_in_bytes());
}

template<Scalar DType, typename Shape_, typename Allocator>
Tensor<DType, Shape_, Allocator>::Tensor(TensorInitializer<DType, Shape_>&& initializer)
: Tensor<DType, Shape_>::Tensor(initializer.values) {}

template<Scalar DType, typename Shape_, typename Allocator>
constexpr DType* Tensor<DType, Shape_, Allocator>::data() noexcept { return data_->data(); }

template<Scalar DType, typename Shape_, typename Allocator>
constexpr const DType* Tensor<DType, Shape_, Allocator>::data() const noexcept { return data_->data(); }
//endregion constructors

template <ExplicitShape NewShape, ExplicitShape OldShape, Scalar DType>
    requires (OldShape::size == NewShape::size)
Tensor<DType, NewShape>&
reshape(Tensor<DType, OldShape>& t) {
    return t;
}

template <ExplicitShape NewShape, ExplicitShape OldShape, Scalar DType>
    requires(DeduceShape<OldShape, NewShape>::deducible)
Tensor<DType, typename DeduceShape<OldShape, NewShape>::Shape>&
reshape(Tensor<DType, OldShape>& t) {
    return t;
}


//region 0D tensor specialization
//region constructors
template<Scalar DType, typename Allocator>
Tensor<DType, Shape<>, Allocator>::Tensor(DType value) : data_(value) {}

template<Scalar DType, typename Allocator>
Tensor<DType, Shape<>, Allocator>::Tensor(TensorInitializer<DType, Shape>&& initializer)
: Tensor<DType, Shape, Allocator>::Tensor(initializer.value) {}
//endregion constructors

template<Scalar DType, typename Allocator>
constexpr DType* Tensor<DType, Shape<>, Allocator>::data() noexcept { return &data_; }

template<Scalar DType, typename Allocator>
constexpr const DType* Tensor<DType, Shape<>, Allocator>::data() const noexcept { return &data_; }

//endregion 0D specialization