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

    using tensorDimension = long;  // 2^32 = 4G, probably don't need larger
    using tensorRank = size_t;
    using tensorSize = size_t;

    template <tensorRank rank_>
    struct Shape {
        std::array<tensorDimension, rank_> dimensions;

        constexpr Shape();
        constexpr ~Shape();

        constexpr Shape(const tensorDimension (&array)[rank_]); // NOLINT(google-explicit-constructor)

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) == rank_)
        constexpr Shape(const Dims& ... dims); // NOLINT(google-explicit-constructor)

        constexpr Shape(const Shape<rank_>& other);  // NOLINT(google-explicit-constructor)

        constexpr Shape<rank_>& operator=(const Shape<rank_>& other);

        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& other) const;

        constexpr tensorDimension operator[](tensorRank i) const;

        [[nodiscard]]
        constexpr tensorRank rank() const;

        [[nodiscard]]
        constexpr tensorSize size() const;

        [[nodiscard]]
        constexpr bool isValidExplicit() const;

        [[nodiscard]]
        constexpr bool isValidImplicit() const;

        [[nodiscard]]
        constexpr bool isValid() const;
    };

    template <>
    struct Shape<0> {

        constexpr Shape() = default;

        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& other) {
            return otherRank == 0;
        };

        [[nodiscard]]
        static constexpr tensorRank rank() { return 0; }

        [[nodiscard]]
        static constexpr tensorSize size() { return 1; };

        [[nodiscard]]
        static constexpr bool isValidExplicit() { return true; };

        [[nodiscard]]
        static constexpr bool isValidImplicit() { return false; };

        [[nodiscard]]
        static constexpr bool isValid() { return true; };
    };

    Shape() -> Shape<0>;
    Shape(tensorDimension) -> Shape<1>;

    template<std::convertible_to<tensorDimension> ... Dims>
    Shape(Dims ... dims) -> Shape<sizeof...(dims)>;

    template<tensorRank N>
    Shape(const tensorDimension (&array)[N]) -> Shape<N>;

    template<tensorRank explicitRank, tensorRank implicitRank>
    constexpr Shape<implicitRank> deduceShape(const Shape<explicitRank>& explicitShape, const Shape<implicitRank>& implicitShape);
}

#endif //TENSOR_SHAPE_PRIVATE_H
