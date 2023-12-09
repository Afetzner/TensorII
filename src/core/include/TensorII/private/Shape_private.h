//
// Created by Amy Fetzner on 12/6/2023.
//

#ifndef TENSOR_SHAPE_PRIVATE_H
#define TENSOR_SHAPE_PRIVATE_H

#include <type_traits>
#include <functional>
#include <numeric>
#include <ranges>
#include <array>
#include <stdexcept>

namespace TensorII::Core {

    using tensorDimension = long long;
    using tensorRank = size_t;
    using tensorSize = size_t;

    template <tensorRank rank_>
    struct Shape {
        static constexpr tensorRank rank = rank_;
        std::array<tensorDimension, rank> dimensions;

        constexpr Shape() requires(rank == 0) = default;

        constexpr Shape(const tensorDimension (&array)[rank]); // NOLINT(google-explicit-constructor)

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) == rank_)
        constexpr Shape(const Dims& ... dims); // NOLINT(google-explicit-constructor)

        constexpr Shape(const Shape<rank_>& other);  // NOLINT(google-explicit-constructor)

        constexpr Shape<rank_>& operator=(const Shape<rank>& other);

        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& other);

        [[nodiscard]]
        constexpr tensorSize size() const;

        [[nodiscard]]
        constexpr bool isValidExplicit() const;

        [[nodiscard]]
        constexpr bool isValidImplicit() const;

        [[nodiscard]]
        constexpr bool isValid() const;
    };

    Shape() -> Shape<0>;
    Shape(tensorDimension) -> Shape<1>;

    template<std::convertible_to<tensorDimension> ... Dims>
    Shape(Dims ... dims) -> Shape<sizeof...(dims)>;

    template<tensorRank N>
    Shape(const tensorDimension (&array)[N]) -> Shape<N>;

    template<tensorRank explicitRank, tensorRank implicitRank>
    constexpr Shape<implicitRank> DeduceShape(const Shape<explicitRank>& explicitShape, const Shape<implicitRank>& implicitShape);
}

#endif //TENSOR_SHAPE_PRIVATE_H
