//
// Created by Amy Fetzner on 12/18/2023.
//

#ifndef TENSOR_INDEXEDTENSORVIEW_TPP
#define TENSOR_INDEXEDTENSORVIEW_TPP

namespace TensorII::Core {

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, size_t index>
    constexpr IndexedTensorView<UnderlyingTensor, apparentShape, index>::IndexedTensorView(
            IndexedTensorView::pointer source)
    {}

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, size_t index>
    constexpr IndexedTensorView<UnderlyingTensor, apparentShape, index>::IndexedTensorView(
            const Tensor<element_type, apparentShape>& source)
    requires(apparentShape == underlyingShape && index == 0)
    {}

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, size_t index>
    constexpr IndexedTensorView<UnderlyingTensor, apparentShape, index>::IndexedTensorView(
            const TensorView<Tensor<element_type, apparentShape>, apparentShape> &source)
    requires(apparentShape == underlyingShape && index == 0)
    {}

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, size_t index>
    constexpr auto IndexedTensorView<UnderlyingTensor, apparentShape, index>::operator[](IndexTriple triple) const {
        return nullptr;
    }
}

#endif //TENSOR_INDEXEDTENSORVIEW_TPP
