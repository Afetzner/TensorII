//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSOR_PREDECL_H
#define TENSOR_TENSOR_PREDECL_H

#include "TensorII/TensorDType.h"
#include "TensorII/Shape.h"
#include "memory"

namespace TensorII::Core {
    using namespace Private;

    template <Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    class Tensor;

    namespace Private {
        template<Scalar DType, auto shape>
        void derived_from_tensor_impl(const Tensor<DType, shape> &);

        template <class T>
        concept derived_from_tensor = requires(const T& t) {
            derived_from_tensor_impl(t);
        };
    }
}

#endif //TENSOR_TENSOR_PREDECL_H
