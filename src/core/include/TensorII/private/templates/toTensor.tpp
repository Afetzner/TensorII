//
// Created by Amy Fetzner on 12/2/2023.
//

#ifndef TENSOR_TOTENSOR_TPP
#define TENSOR_TOTENSOR_TPP

#include "TensorII/Tensor.h"

namespace TensorII::Core {
    //region toTensor
    // 0 dimensions
    template<Scalar DType>
    constexpr Tensor<DType, Shape<0>{}>
    toTensor(const DType value)
    {
        return Tensor<DType, Shape<0>{}>(Private::TensorInitializer<DType, Shape<0>{}>(value));
    }

    // 1 dimension
    template<Scalar DType, tensorDimension dimension>
    constexpr auto toTensor(const DType (&&array)[dimension]) // -> Tensor<DType, Shape<1>{dimension}, Allocator>
    {
        constexpr const Shape shape_ = Shape<1>{dimension};
        return Tensor<DType, shape_>(Private::TensorInitializer<DType, Shape<1>{dimension}>(array));
    }

    // 2 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2>
    constexpr auto toTensor(const DType (&&array)[d1][d2]) // -> Tensor<DType, Shape<2>{d1, d2}, Allocator>
    {
        constexpr const Shape shape_ = Shape < 2 > {d1, d2};
        return Tensor<DType, shape_> (Private::TensorInitializer<DType, shape_>(array));
    }

    // 3 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3>
    constexpr auto toTensor(const DType (&&array)[d1][d2][d3]) // -> Tensor<DType, Shape<3>{d1, d2, d3}, Allocator>
    {
        constexpr const Shape shape_ = Shape <3> {d1, d2, d3};
        return Tensor<DType, shape_> (Private::TensorInitializer<DType, shape_>(array));
    }

    // 4 dimensions
    template<Scalar DType,
            tensorDimension d1,
            tensorDimension d2,
            tensorDimension d3,
            tensorDimension d4>
    constexpr auto toTensor(const DType (&&array)[d1][d2][d3][d4]) // -> Tensor<DType, Shape<4>{d1, d2, d3, d4}, Allocator>
    {
        constexpr const Shape shape_ = Shape <4> {d1, d2, d3, d4};
        return Tensor<DType, shape_> (Private::TensorInitializer<DType, shape_>(array));
    }
    //endregion toTensor
}

#endif //TENSOR_TOTENSOR_TPP
