//
// Created by Amy Fetzner on 11/26/2023.
//

#ifndef TENSOR_TENSORDTYPE_H
#define TENSOR_TENSORDTYPE_H

#include <type_traits>

namespace TensorII::Core {
    template<typename T>
    concept Scalar
            = std::integral<T>
            || std::floating_point<T>;

    template<typename Arr>
    concept ScalarArray
            = std::is_bounded_array_v<Arr>
              && Scalar<typename std::remove_all_extents_t<Arr>>;

}

#endif //TENSOR_TENSORDTYPE_H
