//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSOR_PREDECL_H
#define TENSOR_TENSOR_PREDECL_H

#include "TensorII/TensorDType.h"
#include "memory"

namespace TensorII::Core {
    template <Scalar DType>
    using TensorDefaultAllocator = std::allocator<DType>;

    template <Scalar DType, auto shape_, typename Allocator = TensorDefaultAllocator<DType>>
    class Tensor;
}

#endif //TENSOR_TENSOR_PREDECL_H
