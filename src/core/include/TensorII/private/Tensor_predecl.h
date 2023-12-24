//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSOR_PREDECL_H
#define TENSOR_TENSOR_PREDECL_H

#include "TensorII/TensorDType.h"
#include "memory"

namespace TensorII::Core {
    template <Scalar DType, auto shape_>
    class Tensor;

    namespace Private {
        template<template<Scalar, auto> class Template, Scalar DType, auto shape>
        void derived_from_tensor_specialization_impl(const Template<DType, shape> &);

        template <class T, template <Scalar, auto> class DerivedFromTensor>
        concept derived_from_tensor_specialization_of = requires(const T& t) {
            Private::derived_from_tensor_specialization_impl<DerivedFromTensor>(t);
        };
    }
}

#endif //TENSOR_TENSOR_PREDECL_H
