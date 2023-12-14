//
// Created by Amy Fetzner on 12/12/2023.
//

#ifndef TENSOR_CONCEPTUTIL_H
#define TENSOR_CONCEPTUTIL_H

#include "ranges"

namespace TensorII::Core::Util {

    template<typename R, typename T>
    concept RangeOfSameAs = std::ranges::range<R>
                            && std::same_as<std::ranges::range_value_t<R>, T>;

    template<typename R, typename T>
    concept RangeOfConvertibleTo = std::ranges::range<R>
                                   && std::convertible_to<std::ranges::range_value_t<R>, T>;

    template<typename R, typename T>
    concept ViewOfSameAs = std::ranges::view<R>
                           && std::same_as<std::ranges::range_value_t<R>, T>;

    template<typename R, typename T>
    concept ViewOfConvertibleTo = std::ranges::view<R>
                                  && std::convertible_to<std::ranges::range_value_t<R>, T>;
}
#endif //TENSOR_CONCEPTUTIL_H
