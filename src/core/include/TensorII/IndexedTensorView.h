//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_INDEXEDTENSORVIEW_H
#define TENSOR_INDEXEDTENSORVIEW_H

#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/AnyShape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/private/TensorIndex.h"
#include "TensorII/TensorView.h"

namespace TensorII::Core {
    using namespace Private;

    template<derived_from_tensor_specialization_of<Tensor> UnderlyingTensor, auto apparentShape, size_t index>
    class IndexedTensorView {
    public:
        using element_type = UnderlyingTensor::value_type;
        using value_type = std::remove_cv_t<typename UnderlyingTensor::value_type>;
        using size_type = tensorSize;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        static constexpr Shape underlyingShape = UnderlyingTensor::shape;

    private:
        IndexedTensorView underlyingView;
        std::array<IndexTriple, index> indecies;

        constexpr IndexedTensorView(Tensor<element_type, apparentShape>& source)  // NOLINT(google-explicit-constructor)
        requires(apparentShape == underlyingShape && index == 0);

        constexpr explicit IndexedTensorView(pointer source);

    public:

        static constexpr const tensorRank rank = apparentShape.rank();

        constexpr auto operator[](IndexTriple triple) const;

        AnyShape<rank> shape() { return apparentShape; }
    };
}

#endif //TENSOR_INDEXEDTENSORVIEW_H

#include "private/templates/IndexedTensorView.tpp"