//
// Created by Amy Fetzner on 12/17/2023.
//

#ifndef TENSOR_TYPES_H
#define TENSOR_TYPES_H

namespace TensorII::Core{

    using tensorDimension = long;  // 2^32 = 4G, probably don't need larger
    using tensorIndex = tensorDimension;
    using tensorRank = unsigned long;
    using tensorSize = unsigned long;

    struct from_range_t { explicit from_range_t() = default; };
    constexpr from_range_t from_range = from_range_t{};
}

#endif //TENSOR_TYPES_H
