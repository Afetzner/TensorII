//
// Created by Amy Fetzner on 12/12/2023.
//

#ifndef TENSOR_CONCEPTUTIL_H
#define TENSOR_CONCEPTUTIL_H

#include "ranges"

namespace TensorII::Core::Util {
    template< class R, class T >
    concept ContainerCompatibleRange =
            std::ranges::input_range<R> &&
            std::convertible_to<std::ranges::range_reference_t<R>, T>;

    template< class R, class T >
    concept SizedContainerCompatibleRange =
            std::ranges::input_range<R> &&
            std::ranges::sized_range<R> &&
            std::convertible_to<std::ranges::range_reference_t<R>, T>;

    // See https://stackoverflow.com/a/70130881/22910775
    // Which references https://github.com/microsoft/STL/blob/3c2fd04d441d46ec9d914d9cbb621a3bac96c3a5/stl/inc/xutility#L2684-L2690
namespace Private {
    template<template<class...> class Template, class... Args>
    void derived_from_specialization_impl(const Template<Args...> &);
}
    template <class T, template <class...> class Template>
    concept derived_from_specialization_of = requires(const T& t) {
        Private::derived_from_specialization_impl<Template>(t);
    };

}
#endif //TENSOR_CONCEPTUTIL_H
