//
// Created by Amy Fetzner on 12/10/2023.
//

#ifndef TENSOR_ANYSHAPE_PRIVATE_H
#define TENSOR_ANYSHAPE_PRIVATE_H

#include <type_traits>
#include <optional>
#include <ranges>
#include <array>

#include "TensorII/Shape.h"

namespace TensorII::Core {

    template <tensorRank maxRank>
    class AnyShape{
        std::array<tensorDimension, maxRank> dimensions;
        std::optional<tensorRank> currRank;

    public:
        constexpr ~AnyShape();
        constexpr AnyShape();

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank)
        constexpr explicit AnyShape(const Dims ... dims);

        template <typename Range>
        constexpr explicit AnyShape(Range range)
        requires (std::ranges::range<Range>
                  && std::ranges::sized_range<Range>
                  && std::convertible_to<std::ranges::range_value_t<Range>, tensorDimension>);

        template<typename Range>
        AnyShape& emplace(Range range)
        requires (std::ranges::range<Range>
                  && std::ranges::sized_range<Range>
                  && std::convertible_to<std::ranges::range_value_t<Range>, tensorDimension>);

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank)
        AnyShape& emplace(const Dims ... dims);

        template<tensorRank newRank>
        requires(newRank <= maxRank)
        constexpr Shape<newRank> shape() const;

        template <std::convertible_to<tensorDimension> ... Dims>
        [[nodiscard]]
        constexpr AnyShape<maxRank> augmented(const Dims ... dims) const;

        template<typename Range>
        [[nodiscard]]
        constexpr AnyShape<maxRank> augmented(Range dims) const
        requires (std::ranges::range<Range>
                && std::ranges::sized_range<Range>
                && std::convertible_to<std::ranges::range_value_t<Range>, tensorDimension>);

        [[nodiscard]]
        constexpr AnyShape<maxRank> demoted(tensorRank newRank) const;

        template <tensorRank newRank, std::convertible_to<tensorDimension> ... Dims>
        requires(newRank <= maxRank + sizeof...(Dims))
        void augment(const Dims ... dims);

        template <tensorRank newRank, tensorRank rankDiff>
        requires(newRank <= maxRank + rankDiff)
        void augment(Util::ViewOfConvertibleTo<tensorDimension> auto dims);

        void demote(tensorRank newRank);

        constexpr void reset();

        template <tensorRank otherRank>
        requires(otherRank <= maxRank)
        AnyShape& operator=(const Shape<otherRank>& otherShape);

        template <tensorRank otherMaxRank>
        AnyShape& operator=(const AnyShape<otherMaxRank>& shape);

        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& otherShape) const;

        template <tensorRank otherMaxRank>
        constexpr bool operator==(const AnyShape<otherMaxRank>& otherAnyShape) const;

        constexpr tensorDimension operator[](tensorRank i) const;

        [[nodiscard]]
        constexpr tensorRank rank() const;

        [[nodiscard]]
        constexpr tensorRank size() const;

        [[nodiscard]]
        constexpr bool isValidExplicit() const;

        [[nodiscard]]
        constexpr bool isValidImplicit() const;

        [[nodiscard]]
        constexpr bool isValid() const;
    };
}

#endif //TENSOR_ANYSHAPE_H
