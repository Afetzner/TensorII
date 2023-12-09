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
    Tensor<DType, Shape<0>{}, Allocator>
            toTensor(DType value)
    {
        return Tensor<DType, Shape<0>{}>(Private::TensorInitializer<DType, Shape<0>{}>(value));
    }

    // 1 dimension
    template<Scalar DType,
            tensorDimension dimension,
            typename Allocator = TensorDefaultAllocator<DType>>
    auto toTensor(DType (&&array)[dimension]) // -> Tensor<DType, Shape<1>{dimension}, Allocator>
    {
        return Tensor<DType, Shape<1>{dimension}>(Private::TensorInitializer<DType, Shape<1>{dimension}>(array));
    }

    // 2 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            typename Allocator = TensorDefaultAllocator<DType>>
    auto toTensor( DType (&&array)[d1][d2]) // -> Tensor<DType, Shape<2>{d1, d2}, Allocator>
    {
        constexpr Shape shape = Shape<2>{d1, d2};
        return Tensor<DType, shape> (Private::TensorInitializer<DType, shape>(array));
    }

    // 3 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3,
            typename Allocator = TensorDefaultAllocator<DType>>
    auto toTensor( DType (&&array)[d1][d2][d3]) // -> Tensor<DType, Shape<3>{d1, d2, d3}, Allocator>
    {
        constexpr Shape shape = Shape<3>{d1, d2, d3};
        return Tensor<DType, shape> (Private::TensorInitializer<DType, shape>(array));
    }

    // 4 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3,
            tensorDimension d4,
            typename Allocator = TensorDefaultAllocator<DType>>
    auto toTensor( DType (&&array)[d1][d2][d3][d4]) // -> Tensor<DType, Shape<4>{d1, d2, d3, d4}, Allocator>
    {
        constexpr Shape shape = Shape<4>{d1, d2, d3, d4};
        return Tensor<DType, shape> (Private::TensorInitializer<DType, shape>(array));
    }
    //endregion toTensor
}

#endif //TENSOR_TOTENSOR_H