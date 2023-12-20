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

        constexpr IndexTriple()
                : start(std::nullopt), stop(std::nullopt), step(std::nullopt) {};

        constexpr IndexTriple(tensorIndex start) // NOLINT(google-explicit-constructor)
                : start(start), stop(std::nullopt), step(std::nullopt) {};

        constexpr IndexTriple(tensorIndex start, tensorIndex stop)
                : start(start), stop(stop), step(std::nullopt) {};

        constexpr IndexTriple(tensorIndex start, tensorIndex stop, tensorIndex step)
                : start(start), stop(stop), step(step) {};

        constexpr bool operator==(IndexTriple& other) const {
            return other.start == start and other.stop == stop and other.step == step;
        }

        [[nodiscard]]
        constexpr bool is_empty() const {
            return not start.has_value() and not stop.has_value() and step.has_value();
        };

        [[nodiscard]]
        constexpr bool is_singular() const {
            return start.has_value() and not stop.has_value() and step.has_value();
        };
    };

}

#endif //TENSOR_TENSORINDEX_H
