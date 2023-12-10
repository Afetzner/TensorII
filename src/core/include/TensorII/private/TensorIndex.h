//
// Created by Amy Fetzner on 12/10/2023.
//

#ifndef TENSOR_TENSORINDEX_H
#define TENSOR_TENSORINDEX_H

#include "TensorII/Shape.h"
#include "optional"

namespace TensorII::Core{

    using tensorIndex = tensorDimension;

    struct IndexTriple {
        std::optional<tensorIndex> start;
        std::optional<tensorIndex> stop;
        std::optional<tensorIndex> step;

        constexpr IndexTriple(std::optional<tensorIndex> start = std::nullopt,
                              std::optional<tensorIndex> stop = std::nullopt,
                              std::optional<tensorIndex> step = std::nullopt);
    };

}


#endif //TENSOR_TENSORINDEX_H
