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
}
#endif //TENSOR_CONCEPTUTIL_H
