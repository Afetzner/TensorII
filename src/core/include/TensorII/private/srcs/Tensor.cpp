//
// Created by Amy Fetzner on 11/19/2023.
//

#include "TensorII/private/Tensor_private.h"

using namespace TensorII::Core;

template<Scalar DType, typename Shape_, typename Allocator>
void Tensor<DType, Shape_, Allocator>::ArrayDeleter::operator()(std::array<DType, size()> *a) {
    Allocator allocator;
    allocator.deallocate((DType *)a, size());
}

//region constructors
template<Scalar DType, typename Shape_, typename Allocator>
Tensor<DType, Shape_, Allocator>::Tensor(TensorInitializer<DType, Shape_>::Array& array) {
    static_assert(sizeof(Array) == sizeof(DType) * size(),
            "Mismatch of sizeof(DType[N]) and std::array<DType, N>, cursed compiler?");
    Allocator allocator;
    Array* array_p = new(allocator.allocate(size())) Array;
    data_ = std::unique_ptr<Array, ArrayDeleter>(array_p);
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

//region reshape
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
//endregion reshape

//region indexing
template<Scalar DType, typename Shape_, typename Allocator>
template<tensorRank rank>
const DType* Tensor<DType, Shape_, Allocator>::operator[](const TensorIndexer<rank>& indexer) const {
    return data_->data();
}
template<Scalar DType, typename Shape_, typename Allocator>
template<tensorRank rank>
DType* Tensor<DType, Shape_, Allocator>::operator[](const TensorIndexer<rank>& indexer) {
    return data_->data();
}

template<Scalar DType, typename Shape_, typename Allocator>
template<tensorRank rank>
const DType* Tensor<DType, Shape_, Allocator>::operator[](const tensorIndex (&indecies)[rank]) const {
    return operator[](TensorIndexer<rank>{indecies});
}
template<Scalar DType, typename Shape_, typename Allocator>
template<tensorRank rank>
DType* Tensor<DType, Shape_, Allocator>::operator[](const tensorIndex (&indecies)[rank]){
    return operator[](TensorIndexer<rank>{indecies});
}
//endregion indexing

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
