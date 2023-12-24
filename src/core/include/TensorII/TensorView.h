//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSORVIEW_H
#define TENSOR_TENSORVIEW_H

#include <ranges>

#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/AnyShape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/private/TensorIndex.h"

namespace TensorII::Core {
    using namespace Private;

    template<derived_from_tensor_specialization_of<Tensor> UnderlyingTensor, auto apparentShape, size_t index>
    class IndexedTensorView;

    template<derived_from_tensor_specialization_of<Tensor> UnderlyingTensor, auto apparentShape>
    class TensorView {
    public:
        using element_type = UnderlyingTensor::value_type;
        using value_type = std::remove_cv_t<typename UnderlyingTensor::value_type>;
        using size_type = tensorSize;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

    private:
        element_type* source;
        constexpr explicit TensorView(element_type* source);

    public:
        static constexpr const tensorRank rank = apparentShape.rank();

        constexpr TensorView(Tensor<element_type, apparentShape>& source)  // NOLINT(google-explicit-constructor)
        requires(apparentShape == source.shape());

        constexpr TensorView<UnderlyingTensor, Shape<rank - 1>(from_range, apparentShape | std::ranges::views::drop(1))>
        operator[](tensorIndex idx) const;

        constexpr auto // IndexedTensorView<UnderlyingTensor, apparentShape, 1>
        operator[](IndexTriple triple) const;

        AnyShape<rank> shape() { return apparentShape; }
    };

    template<Scalar DType, auto shape>
    TensorView(Tensor<DType, shape>) -> TensorView<Tensor<DType, shape>, shape>;
}

#endif //TENSOR_TENSORVIEW_H

#include "private/templates/TensorView.tpp"