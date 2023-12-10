//
// Created by Amy Fetzner on 12/10/2023.
//

#ifndef TENSOR_ANYSHAPE_PRIVATE_H
#define TENSOR_ANYSHAPE_PRIVATE_H

#include <type_traits>
#include <optional>
#include "TensorII/Shape.h"


namespace TensorII::Core {

    template <tensorRank maxRank>
    class AnyShape{
        std::aligned_storage_t<sizeof(Shape<maxRank>), alignof(Shape<maxRank>)> data;
        std::optional<tensorRank> currRank;

    public:
        constexpr ~AnyShape();
        constexpr AnyShape();

        template<tensorRank rank>
        requires(rank <= maxRank)
        constexpr AnyShape& emplace(const tensorDimension (&array)[rank]);

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) < maxRank)
        constexpr AnyShape& emplace(Dims ... dims);

        template<tensorRank newRank>
        requires(newRank <= maxRank)
        constexpr const Shape<newRank>* shape() const;

        constexpr void reset();

        template <tensorRank otherRank>
        requires(otherRank <= maxRank)
        constexpr AnyShape& operator=(const Shape<otherRank>& otherShape);

        template <tensorRank otherMaxRank>
        constexpr AnyShape& operator=(const AnyShape<otherMaxRank>& shape);

        template <tensorRank otherRank>
        requires(otherRank <= maxRank)
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
