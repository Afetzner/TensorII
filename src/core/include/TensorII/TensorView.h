//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSORVIEW_H
#define TENSOR_TENSORVIEW_H

#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/AnyShape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/private/TensorIndex.h"

namespace TensorII::Core {
    template<Scalar DType, auto underlyingShape>
    class TensorView {
        static constexpr tensorRank maxRank = underlyingShape.rank();
        Tensor<DType, underlyingShape>& source;
        AnyShape<maxRank> apparentShape;
        const tensorRank index;
        std::array<IndexTriple, maxRank> indecies;

        constexpr TensorView(const TensorView& other, IndexTriple triple);

    public:
        constexpr TensorView(Tensor<DType, underlyingShape>& source); // NOLINT(google-explicit-constructor)

        template <std::convertible_to<IndexTriple> ... Triples>
        requires (sizeof...(Triples) <= underlyingShape.rank())
        constexpr explicit TensorView(Tensor<DType, underlyingShape>& source, const Triples& ... triples);

        template <typename Range>
        requires (Util::SizedContainerCompatibleRange<Range, IndexTriple>)
        constexpr TensorView(Tensor<DType, underlyingShape>& source, from_range_t, Range&& range)
        requires (std::ranges::size(range) <= underlyingShape.rank());

        constexpr auto operator[](IndexTriple triple) const;

        const AnyShape<maxRank>& shape() { return apparentShape; }
    };

    template<Scalar DType, auto shape>
    TensorView(Tensor<DType, shape>) -> TensorView<DType, shape>;
}

#endif //TENSOR_TENSORVIEW_H

#include "private/srcs/TensorView.tpp"