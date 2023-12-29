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
    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
    requires (derived_from_static_shape<decltype(apparentShape)>)constexpr
    TensorView<UnderlyingTensor, apparentShape, index_>::TensorView(element_type *source)
            : source(source)
            , indecies {}
    {}

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
    requires (derived_from_static_shape<decltype(apparentShape)>)
    template<Util::ContainerCompatibleRange<TensorIndex> Range>
    constexpr TensorView<UnderlyingTensor, apparentShape, index_>::TensorView(element_type *source,
                                                                              const Range &&range)
            : source(source)
            , indecies {}
    {
        std::ranges::copy_n(range.begin(), index_, indecies.begin());
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
    requires (derived_from_static_shape<decltype(apparentShape)>)
    template<Util::ContainerCompatibleRange<TensorIndex> Range>
    constexpr TensorView<UnderlyingTensor, apparentShape, index_>::TensorView(element_type *source,
                                                                              const Range &&range,
                                                                              TensorIndex index)
            : source(source)
            , indecies {}
    {
        std::ranges::copy_n(range.begin(), index_ - 1, indecies.begin());
        indecies[index_ - 1] = index;
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, apparentShape, index_>::
    TensorView(Tensor<element_type, apparentShape> &source)
    : source (source.data())
    , indecies {}
    {
        if (apparentShape == source.shape()){
            // I should probably use a more specific error type here
            throw std::logic_error("Creating a tensor view from a tensor who's shape does not match");
        }
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, ShapeTail(apparentShape), index_>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    operator[](Single single) {
        constexpr auto restShape = ShapeTail(apparentShape);
        using NewTensorView = TensorView<UnderlyingTensor, restShape, index_>;
        element_type* newSource = &source[single.single() * restShape.n_elems()];
        return NewTensorView{newSource,
                             indecies| std::ranges::views::take(index_)};
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, apparentShape, index_ + 1>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    operator[](Empty empty) {
        using NewTensorView = TensorView<UnderlyingTensor, apparentShape, index_ + 1>;
        return NewTensorView {source,
                              indecies| std::ranges::views::take(index_),
                              static_cast<TensorIndex>(empty)};
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    constexpr TensorView<UnderlyingTensor, IndeterminateShape<TensorView<UnderlyingTensor, apparentShape, index_>::rank()>, index_ + 1>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    operator[](Triple triple) {
        // TODO
//        using NewTensorView = TensorView<UnderlyingTensor, DynamicShape<rank()>(), index_ + 1>;
//        return NewTensorView {source,
//                              indecies | std::ranges::views::take(index_),
//                              static_cast<TensorIndex>(triple)};
        return {};
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
        requires (derived_from_static_shape<decltype(apparentShape)>)
    template<Triple triple>
    constexpr auto //TensorView<UnderlyingTensor, apparentShape.replace(index_, triple.count(apparentShape[0])), index_ + 1>
    TensorView<UnderlyingTensor, apparentShape, index_>::
    slice()
    {
        constexpr auto newShape = apparentShape.replace(index_, triple.count(apparentShape[0]));
        using NewTensorView = TensorView<UnderlyingTensor, newShape, index_ + 1>;
        return NewTensorView {source,
                              indecies | std::ranges::views::take(index_),
                              static_cast<TensorIndex>(triple)};
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
