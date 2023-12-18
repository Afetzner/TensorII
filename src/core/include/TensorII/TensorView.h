//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSORVIEW_H
#define TENSOR_TENSORVIEW_H

#include "TensorII/private/headers/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/AnyShape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/private/TensorIndex.h"

namespace TensorII::Core {
    template<Scalar DType, auto shape, auto underlyingShape>
    class TensorView {
        Tensor<DType, underlyingShape>* source;
        std::array<IndexTriple, shape.rank()> indecies;

    public:
        constexpr TensorView(Tensor<DType, underlyingShape>* source) : source(source) {};

        void operator[](IndexTriple triple);
    };
}

#endif //TENSOR_TENSORVIEW_H
