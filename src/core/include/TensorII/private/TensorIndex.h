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

        constexpr explicit IndexTriple()
                : start(std::nullopt), stop(std::nullopt), step(std::nullopt) {};

        constexpr explicit IndexTriple(tensorIndex start)
                : start(start), stop(std::nullopt), step(std::nullopt) {};

        constexpr explicit IndexTriple(tensorIndex start, tensorIndex stop)
                : start(start), stop(stop), step(std::nullopt) {};

        constexpr explicit IndexTriple(tensorIndex start, tensorIndex stop, tensorIndex step)
                : start(start), stop(stop), step(step) {};
    };

}

#endif //TENSOR_TENSORINDEX_H
