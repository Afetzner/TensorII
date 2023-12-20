//
// Created by Amy Fetzner on 12/18/2023.
//

#ifndef TENSOR_TENSORVIEW_TPP
#define TENSOR_TENSORVIEW_TPP

#include "TensorII/TensorView.h"

#include <ranges>

#include "TensorII/TensorDType.h"

namespace TensorII::Core {

    template<Scalar DType, auto underlyingShape>
    constexpr TensorView<DType, underlyingShape>::TensorView(Tensor<DType, underlyingShape>& source)
        : source(source)
        , apparentShape(source.shape())
        , index(0)
        , indecies{0}
    {}

    template<Scalar DType, auto underlyingShape>
    template<std::convertible_to<IndexTriple> ... Triples>
    requires (sizeof...(Triples) <= underlyingShape.rank())
    constexpr TensorView<DType, underlyingShape>::TensorView(Tensor<DType, underlyingShape>& source, const Triples& ... triples)
        : source(source)
        , index(sizeof...(Triples))
        , indecies{triples ...}
    {
        // TODO: calculate apparent shape
    }

    template<Scalar DType, auto underlyingShape>
    template <typename Range>
    requires (Util::SizedContainerCompatibleRange<Range, IndexTriple>)
    constexpr TensorView<DType, underlyingShape>::TensorView(Tensor<DType, underlyingShape>& source, from_range_t, Range&& range)
    requires (std::ranges::size(range) <= underlyingShape.rank())
        : source(source)
        , index(std::ranges::size(range)) {
        size_t how_many = std::max(std::ranges::size(range), maxRank);
        std::ranges::copy_n(range.begin(), how_many, indecies.begin());
    }

    template<Scalar DType, auto underlyingShape>
    constexpr auto TensorView<DType, underlyingShape>::operator[](IndexTriple triple) const {
        if (triple.is_empty()){ // Index like shape[{}],
            TensorView<DType, underlyingShape> result(source);
            // TODO
            return result;
        }

        else if (triple.is_singular()){ // Index like shape[N],
            AnyShape newShape = AnyShape<maxRank>(from_range, apparentShape | std::views::drop(1));
            TensorView<DType, underlyingShape> result(source);
            // TODO
            return result;
        }

        // Other shape
        TensorView<DType, underlyingShape> result(source);
        // TODO
        return result;
    }
}

#endif //TENSOR_TENSORVIEW_TPP
