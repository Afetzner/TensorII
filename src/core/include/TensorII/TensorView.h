//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSORVIEW_H
#define TENSOR_TENSORVIEW_H

#include "TensorII/private/headers/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/private/TensorIndex.h"

namespace TensorII::Core {
    template<Scalar DType, auto underlyingShape>
    class TensorView {
        Tensor<DType, underlyingShape>* source;
        std::array<IndexTriple, underlyingShape.rank()> indecies;

//        Shape shape;

    public:
        constexpr TensorView()

        consteval tensorRank rank();

        consteval tensorSize size();

        consteval Tensor<DType, underlyingShape>* tensor();
    };
}

#endif //TENSOR_TENSORVIEW_H
