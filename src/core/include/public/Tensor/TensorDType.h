//
// Created by Amy Fetzner on 11/26/2023.
//

#ifndef TENSOR_TENSORDTYPE_H
#define TENSOR_TENSORDTYPE_H

template<typename T>
concept TensorDType = requires {
    std::integral<T> || std::floating_point<T>;
};

#endif //TENSOR_TENSORDTYPE_H
