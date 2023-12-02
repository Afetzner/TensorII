//
// Created by Amy Fetzner on 12/2/2023.
//

#ifndef TENSOR_TOTENSOR_H
#define TENSOR_TOTENSOR_H

#include "Tensor_private.h"

namespace TensorII::Core {
    //region toTensor
    // 0 dimensions
    template<Scalar DType, typename Allocator = TensorDefaultAllocator<DType>>
    Tensor<DType, Shape<>, Allocator>
            toTensor(DType value)
    {
        return Tensor<DType, Shape<>>(TensorInitializer<DType, Shape<>>(value));
    }

    // 1 dimension
    template<Scalar DType,
            tensorDimension dimension,
            typename Allocator = TensorDefaultAllocator<DType>>
    Tensor<DType, Shape<dimension>, Allocator>
            toTensor(DType (&&array)[dimension])
    {
    return Tensor<DType, Shape<dimension>>(TensorInitializer<DType, Shape<dimension>>(array));
    }

    // 2 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            typename Allocator = TensorDefaultAllocator<DType>>
    Tensor<DType, Shape<d1, d2>, Allocator>
            toTensor( DType (&&array)[d1][d2])
    {
    using Shape = Shape<d1, d2>;
    return Tensor<DType, Shape> (TensorInitializer<DType, Shape>(array));
    }

    // 3 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3,
            typename Allocator = TensorDefaultAllocator<DType>>
    Tensor<DType, Shape<d1, d2, d3>, Allocator>
            toTensor( DType (&&array)[d1][d2][d3])
    {
    using Shape = Shape<d1, d2, d3>;
    return Tensor<DType, Shape> (TensorInitializer<DType, Shape>(array));
    }

    // 4 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3,
            tensorDimension d4,
            typename Allocator = TensorDefaultAllocator<DType>>
    Tensor<DType, Shape<d1, d2, d3, d4>, Allocator>
            toTensor( DType (&&array)[d1][d2][d3][d4])
    {
    using Shape = Shape<d1, d2, d3, d4>;
    return Tensor<DType, Shape> (TensorInitializer<DType, Shape>(array));
    }
    //endregion toTensor
}

#endif //TENSOR_TOTENSOR_H
