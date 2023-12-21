//
// Created by Amy Fetzner on 12/10/2023.
//

#ifndef TENSOR_TENSORINDEX_H
#define TENSOR_TENSORINDEX_H

#include "TensorII/Shape.h"
#include "optional"

namespace TensorII::Core{

    using tensorIndex = tensorDimension;

    class IndexTriple {
        std::optional<tensorIndex> single_;
        std::optional<tensorIndex> start_;
        std::optional<tensorIndex> stop_;
        std::optional<tensorIndex> step_;

    public:
        constexpr IndexTriple(tensorIndex index) // NOLINT(google-explicit-constructor)
        : single_(index)
        , start_(std::nullopt)
        , stop_(std::nullopt)
        , step_(std::nullopt) {};

        constexpr IndexTriple(std::optional<tensorIndex> single = std::nullopt,
                              std::optional<tensorIndex> start = std::nullopt,
                              std::optional<tensorIndex> step = std::nullopt)
                : single_(std::nullopt)
                , start_(single)
                , stop_(start)
                , step_(step) {};

        constexpr bool operator==(IndexTriple& other) const {
            if (single_.has_value()){
                return other.single_ == single_;
            }
            return other.start_ == start_
                   && other.stop_ == stop_
                   && other.step_ == step_;
        }

        inline constexpr const std::optional<tensorIndex>& single() { return single_; }
        inline constexpr const std::optional<tensorIndex>& start()  { return start_; }
        inline constexpr const std::optional<tensorIndex>& step()   { return step_; }
        inline constexpr const std::optional<tensorIndex>& stop()   { return stop_; }

        [[nodiscard]]
        constexpr bool is_empty() const {
            return !single_.has_value()
                && !start_.has_value()
                && !stop_.has_value()
                && !step_.has_value();
        };

        [[nodiscard]]
        constexpr bool is_singular() const {
            return single_.has_value();
        };
    };

}

#endif //TENSOR_TENSORINDEX_H
