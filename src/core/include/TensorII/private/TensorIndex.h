//
// Created by Amy Fetzner on 12/10/2023.
//

#ifndef TENSOR_TENSORINDEX_H
#define TENSOR_TENSORINDEX_H

#include "TensorII/Shape.h"
#include "TensorII/Types.h"
#include <optional>
#include <variant>

namespace TensorII::Core::Indexing {

    class TensorIndex {
    protected:
        struct IndexTriple {
            // Cannot use std::optional since it's not structural
            // Needs to be structural to use it as a literal type as a non-type template parameter
            tensorIndex start;
            tensorIndex stop;
            tensorIndex step;
            bool has_single : 1;
            bool has_start : 1;
            bool has_stop : 1;
            bool has_step : 1;
        };

    public:
        IndexTriple triple;

        constexpr TensorIndex(tensorIndex idx) // NOLINT(google-explicit-constructor)
        : triple{.start = idx,
                .stop = 0,
                .step = 0,
                .has_single = true,
                .has_start = false,
                .has_stop = false,
                .has_step = false}
        {};

        constexpr TensorIndex(std::optional<tensorIndex> start = std::nullopt,  // NOLINT(google-explicit-constructor)
                              std::optional<tensorIndex> stop  = std::nullopt,
                              std::optional<tensorIndex> step  = std::nullopt)
        : triple{.start = start.value_or(0),
                .stop = stop.value_or(0),
                .step = step.value_or(0),
                .has_single = false,
                .has_start = start.has_value(),
                .has_stop = stop.has_value(),
                .has_step = step.has_value()}
        {};

        [[nodiscard]]
        inline constexpr bool is_empty() const {
            return !triple.has_single and !triple.has_start and !triple.has_step and !triple.has_stop;
        }

        [[nodiscard]]
        inline constexpr bool is_singular() const { return triple.has_single; };

        [[nodiscard]]
        inline constexpr bool is_triple() const { return !triple.has_single and
                (triple.has_start or triple.has_step or triple.has_stop); };

        [[nodiscard]]
        inline constexpr std::optional<tensorIndex> single() const {
            if (triple.has_single) { return {triple.start}; }
            return {std::nullopt};
        }

        [[nodiscard]]
        inline constexpr std::optional<tensorIndex> start() const {
            if (triple.has_start){ return {triple.start}; }
            return {std::nullopt};
        }

        [[nodiscard]]
        inline constexpr std::optional<tensorIndex> step() const{
            if (triple.has_step){ return {triple.step}; }
            return {std::nullopt};
        }

        [[nodiscard]]
        inline constexpr std::optional<tensorIndex> stop() const {
            if (triple.has_stop){ return {triple.stop}; }
            return {std::nullopt};
        }
    };

    // HIDING OF BASE'S FUNCTION IN FOLLOWING IS DELIBERATE
    struct Triple : public TensorIndex {
        constexpr Triple() = delete;

        constexpr Triple( // NOLINT(google-explicit-constructor)
                std::optional<tensorIndex> start,
                std::optional<tensorIndex> stop = std::nullopt,
                std::optional<tensorIndex> step = std::nullopt)
                : TensorIndex(start, stop, step)
        {}

        [[nodiscard]] inline constexpr bool is_empty() const {
            return !triple.has_start and !triple.has_step and !triple.has_stop;
        }

        [[nodiscard]] static inline constexpr bool is_singular() { return false; }

        using TensorIndex::start, TensorIndex::stop, TensorIndex::step;

        [[nodiscard]]
        constexpr tensorIndex count(tensorDimension dimLength) const {
            return (stop().value_or(dimLength) - start().value_or(0)) / step().value_or(1);
        }
    };

    struct Single : public TensorIndex {
        constexpr Single() = delete;

        constexpr Single(tensorIndex idx) : TensorIndex(idx) {}; // NOLINT(google-explicit-constructor)

        [[nodiscard]] static inline constexpr bool is_empty() { return false; }

        [[nodiscard]] static inline constexpr bool is_singular() { return true; }

        [[nodiscard]] inline constexpr tensorIndex single() const { return triple.start; }
    };

    struct Empty : public TensorIndex {
        constexpr Empty() = default;

        [[nodiscard]] static inline constexpr bool is_empty() { return true; }

        [[nodiscard]] static inline constexpr bool is_singular() { return false; }

    };

    static_assert(std::is_default_constructible_v<Empty>);
    static_assert(!std::is_default_constructible_v<Single>);
    static_assert(!std::is_default_constructible_v<Triple>);
}

#endif //TENSOR_TENSORINDEX_H
