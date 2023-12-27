//
// Created by Amy Fetzner on 12/18/2023.
//

#ifndef TENSOR_TENSORVIEW_TPP
#define TENSOR_TENSORVIEW_TPP

#include "TensorII/TensorView.h"
#include <ranges>
#include "TensorII/TensorDType.h"

namespace TensorII::Core {
    using namespace Indexing;
    //region Shape apparentShape specialization

    // TensorView(TensorView<Shape{...}, Shape{a, b, c, ...}>, IndexTriple{N})
    // -> TensorView<Shape{...}, Shape{b, c, ...}>
    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    template <auto otherApparentShape>
        requires (ShapeTail(otherApparentShape) == apparentShape)
    constexpr TensorView<UnderlyingTensor, apparentShape, index_>::
    TensorView(TensorView<UnderlyingTensor, otherApparentShape, index_> other, Single single) {
        constexpr Shape ShapeRemaining = ShapeTail(apparentShape);
        size_t true_idx = single.single().value() * ShapeRemaining.n_elems();
        source = &other.source[true_idx];
        indecies = {from_range, other.indecies};
    }

    // TensorView(TensorView<Shape{...}, Shape{a, b, c, ...}>, IndexTriple{})
    // -> TensorView<Shape{...}, Shape{a, b, c, ...}>
    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    template <auto otherApparentShape>
        requires (otherApparentShape == apparentShape)
    constexpr  TensorView<UnderlyingTensor, apparentShape, index_>::
    TensorView(TensorView<UnderlyingTensor, otherApparentShape, index_ - 1> other, Empty empty)
    : source(other.source)
    , indecies{from_range, other.indecies}
    {
        indecies[index_] = empty;
    }

    // TensorView(TensorView<Shape{...}, Shape{a, b, c, ...}>, IndexTriple{x, y, z})
    // -> TensorView<Shape{...}, Shape{(y-z)/z, b, c, ...}>
    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    template <auto otherApparentShape>
        requires (ShapeTail(otherApparentShape) == apparentShape)
    constexpr TensorView<UnderlyingTensor, apparentShape, index_>::
    TensorView(TensorView<UnderlyingTensor, otherApparentShape, index_ - 1> other, Triple triple)
    : source(other.source)
    , indecies{from_range, other.indecies}
    {
        indecies[index_] = triple;
    }


    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, apparentShape, index_>::
    TensorView(Tensor<element_type, apparentShape> &source)
    requires(apparentShape == source.shape())
    : source (source.data())
    , indecies {}
    {
        // TODO
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, ShapeTail(apparentShape), index_ + 1>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    operator[](Single single) {
        using NewTensorView = TensorView<UnderlyingTensor, ShapeTail(apparentShape), index_ + 1>;
        return NewTensorView{*this, single};
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, apparentShape, index_ + 1>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    operator[](Empty empty) {
        using NewTensorView = TensorView<UnderlyingTensor, apparentShape, index_ + 1>;
        return NewTensorView{*this, empty};
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, DynamicShape<TensorView<UnderlyingTensor, apparentShape, index_>::rank()>, index_ + 1>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    operator[](Triple index) {
        // TODO
        using NewTensorView = TensorView<UnderlyingTensor, DynamicShape<rank()>(), index_ + 1>;
        return NewTensorView (*this, index);
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    template<Triple triple>
    constexpr auto //TensorView<UnderlyingTensor, apparentShape.replace(index_, triple.count(apparentShape[0])), index_ + 1>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    slice()
    {
        static constexpr auto newShape = apparentShape.replace(index_, triple.count(apparentShape[0]));
        using NewTensorView = TensorView<UnderlyingTensor, newShape, index_ + 1>;
        return NewTensorView{*this, triple};
    }

//    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
//    requires (derived_from_static_shape<decltype(apparentShape)>)constexpr TensorView<UnderlyingTensor, DynamicShape<TensorView<UnderlyingTensor, apparentShape, index_>::rank()>,
//            index_ + 1> TensorView<UnderlyingTensor, apparentShape, index_>::operator[](TensorIndex index) requires (
//    !std::is_same_v<decltype(index), Single> && !std::is_same_v<decltype(index), Empty>) {
//        return TensorView<UnderlyingTensor, DynamicShape<rank()>, index_ + 1>();
//    }

    //endregion Shape apparentShape specialization


    //region AnyShape apparentShape specialization


    //endregion AnyShape apparentShape specialization
}

#endif //TENSOR_TENSORVIEW_TPP
