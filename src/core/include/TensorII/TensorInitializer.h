//
// Created by Amy Fetzner on 11/24/2023.
//
#ifndef TENSOR_TENSORINITIALIZER_H
#define TENSOR_TENSORINITIALIZER_H

#include <concepts>
#include <type_traits>
#include "Shape.h"
#include "TensorDType.h"

namespace TensorII::Core{

    template<Scalar DType, typename Shape_>
    struct TensorInitializer;

    //region TensorInitializer Definitions
    // 0 dimensions
    template<Scalar DType>
    struct TensorInitializer<DType, Shape<>> {
        DType value;
        TensorInitializer(DType value) // NOLINT(google-explicit-constructor)
        : value(value) {};
    };


    // 1 dimension
    template<Scalar DType, tensorDimension dimension>
    struct TensorInitializer<DType, Shape<dimension>> {
        using Array = DType const [dimension];
        Array& values;

        explicit TensorInitializer(Array& array) : values(array) {}
    };

    // >1 dimension
    template<Scalar DType, tensorDimension dimension, tensorDimension ... rest>
    struct TensorInitializer<DType, Shape<dimension, rest...>> {
    private :
        using LowerArray = typename TensorInitializer<DType, Shape<rest...>>::Array;
    public:
        using Array = LowerArray const [dimension];
        Array& values;

        TensorInitializer(Array& values) // NOLINT(google-explicit-constructor)
        : values(values) {};
    };

//namespace Private {
//    template <tensorDimension, typename>
//    struct AppendDimension;
//
//    template <tensorDimension dimension, tensorDimension ... rest>
//    struct AppendDimension<dimension, Shape<rest ...>> {
//        using Shape = Shape<dimension, rest ...>;
//    };
//
//    template <typename Array>
//    struct ArrayShape;
//
//    template <Scalar DType_>
//    struct ArrayShape<DType_> {
//        using DType = DType_;
//        using Shape = Shape<>;
//    };
//
//    template <typename T, tensorDimension N>
//    struct ArrayShape<T[N]> {
//        using DType = T::DType;
//        using Shape = AppendDimension<N, typename T::Shape>::Shape;
//    };
//}
    //endregion TensorInitializer Definitions

    template<Scalar DType, tensorDimension dimension>
    TensorInitializer(DType(&) [dimension])
    -> TensorInitializer<DType, Shape<dimension>>;

    template<Scalar DType>
    TensorInitializer(DType array[])
    -> TensorInitializer<DType, Shape<sizeof(decltype(array)) / sizeof(DType)>>;
}

#endif //TENSOR_TENSORINITIALIZER_H
