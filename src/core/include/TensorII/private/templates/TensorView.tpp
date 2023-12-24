//
// Created by Amy Fetzner on 12/18/2023.
//

#ifndef TENSOR_TENSORVIEW_TPP
#define TENSOR_TENSORVIEW_TPP

#include "TensorII/TensorView.h"
#include <ranges>
#include "TensorII/TensorDType.h"
#include "TensorII/IndexedTensorView.h"

namespace TensorII::Core {

    template<derived_from_tensor_specialization_of<Tensor> UnderlyingTensor, auto apparentShape>
    constexpr TensorView<UnderlyingTensor, apparentShape>::TensorView(element_type* source)
        : source(source)
    {}

    template<derived_from_tensor_specialization_of<Tensor> UnderlyingTensor, auto apparentShape>
    constexpr TensorView<UnderlyingTensor, apparentShape>::TensorView(Tensor<element_type, apparentShape> &source)
    requires(apparentShape == source.shape())
        : source(source.data())
    {}

    template<derived_from_tensor_specialization_of<Tensor> UnderlyingTensor, auto apparentShape>
    constexpr TensorView<UnderlyingTensor, Shape(from_range, apparentShape | std::ranges::views::drop(1))>
    TensorView<UnderlyingTensor, apparentShape>::operator[](tensorIndex idx) const
    {
        constexpr Shape ShapeRemaining = {from_range, apparentShape | std::ranges::views::drop(1)};
        using Result = TensorView<UnderlyingTensor, ShapeRemaining>;
        size_t true_idx = idx * ShapeRemaining.n_elems;
        return Result(&source[true_idx]);
    }

    template<derived_from_tensor_specialization_of<Tensor> UnderlyingTensor, auto apparentShape>
    constexpr auto //IndexedTensorView<UnderlyingTensor, apparentShape, 1>
    TensorView<UnderlyingTensor, apparentShape>::operator[](IndexTriple triple) const {
        // Index like tensor[N],
        if (triple.is_singular()) {

        }

        // Index like tensor[{}],
        else if (triple.is_empty()) {
            constexpr Shape newShape = apparentShape;

            IndexedTensorView<UnderlyingTensor, newShape, 1>();
        }

        // Other index, like tensor[{1}], tensor[{1, 2}], tensor[{1, 2, 3}]



    }
}




//    template<Scalar DType, auto apparentShape, auto underlyingShape>
//    constexpr TensorView<DType, apparentShape, underlyingShape>::TensorView(const TensorView &other, IndexTriple triple)
//    : source(other.source)
//    , index(other.index + 1)
//    , indecies{}
//    {
//        // Index like shape[{}],
//        if (triple.is_empty()){
//            apparentShape = {AnyShape(other.apparentShape)};
//            std::ranges::copy_n(other.indecies.begin(), other.index, indecies.begin());
//            indecies[index + 1] = triple;
//        }
//
//        // Index like shape[N],
//        else if (triple.is_singular()){
//            apparentShape = {from_range, other.apparentShape | std::views::drop(1)};
//            // indecies = { triple, other.indecies[0], other.indecies[1], other.indecies[2], ... }
//            indecies = {triple};
//            std::ranges::copy_n(other.indecies.begin(), other.index, indecies.begin() + 1);
//        }
//
//        // Other index, like shape[{1}], shape[{1, 2}], shape[{1, 2, 3}]
//        else {
//            // apparentShape = { (stop - start)/step, other.appShape[1], other.appShape[2], other.appShape[3], ... }
//            tensorDimension first_dimension = (triple.stop().value() - triple.start().value()) / triple.step().value();
//            apparentShape = {first_dimension};
//            std::ranges::copy_n(other.apparentShape.begin() + 1, other.index - 1, apparentShape.begin() + 1);
//            // indecies = { triple, other.indecies[0], other.indecies[1], other.indecies[2], ... }
//            indecies = {triple};
//            std::ranges::copy_n(other.indecies.begin(), other.index, indecies.begin() + 1);
//        }
//    }

#endif //TENSOR_TENSORVIEW_TPP
