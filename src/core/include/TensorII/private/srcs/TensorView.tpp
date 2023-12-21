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
    constexpr TensorView<DType, underlyingShape>::TensorView(const TensorView &other, IndexTriple triple)
    : source(other.source)
    , index(other.index + 1)
    , indecies{}
    {
        // Index like shape[{}],
        if (triple.is_empty()){
            apparentShape = {AnyShape(other.apparentShape)};
            std::ranges::copy_n(other.indecies.begin(), other.index, indecies.begin());
            indecies[index + 1] = triple;
        }

        // Index like shape[N],
        else if (triple.is_singular()){
            apparentShape = {from_range, other.apparentShape | std::views::drop(1)};
            // indecies = { triple, other.indecies[0], other.indecies[1], other.indecies[2], ... }
            indecies = {triple};
            std::ranges::copy_n(other.indecies.begin(), other.index, indecies.begin() + 1);
        }

        // Other index, like shape[{1}], shape[{1, 2}], shape[{1, 2, 3}]
        else {
            // apparentShape = { (stop - start)/step, other.appShape[1], other.appShape[2], other.appShape[3], ... }
            tensorDimension first_dimension = (triple.stop().value() - triple.start().value()) / triple.step().value();
            apparentShape = {first_dimension};
            std::ranges::copy_n(other.apparentShape.begin() + 1, other.index - 1, apparentShape.begin() + 1);
            // indecies = { triple, other.indecies[0], other.indecies[1], other.indecies[2], ... }
            indecies = {triple};
            std::ranges::copy_n(other.indecies.begin(), other.index, indecies.begin() + 1);
        }
    }

    template<Scalar DType, auto underlyingShape>
    constexpr TensorView<DType, underlyingShape>::TensorView(Tensor<DType, underlyingShape>& source)
        : source(source)
        , apparentShape(source.shape())
        , index(0)
        , indecies{}
    {}

    template<Scalar DType, auto underlyingShape>
    template<std::convertible_to<IndexTriple> ... Triples>
    requires (sizeof...(Triples) <= underlyingShape.rank())
    constexpr TensorView<DType, underlyingShape>::TensorView(Tensor<DType, underlyingShape>& source, const Triples& ... triples)
        : source(source)
        , index(sizeof...(Triples))
        , indecies{triples ...}
    {
        //TODO
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
        return TensorView(*this, triple);
    }
}

#endif //TENSOR_TENSORVIEW_TPP
