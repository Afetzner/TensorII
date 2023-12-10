//
// Created by Amy Fetzner on 12/10/2023.
//

#include <any>
#include "TensorII/private/headers/AnyShape_private.h"

namespace TensorII::Core {

    template<tensorRank maxRank>
    constexpr AnyShape<maxRank>::AnyShape()
    : currRank(std::nullopt)
    {}

    template<tensorRank maxRank>
    constexpr AnyShape<maxRank>::~AnyShape() = default;

    template<tensorRank maxRank>
    template<tensorRank rank_>
    requires(rank_ <= maxRank)
    constexpr AnyShape<maxRank>& AnyShape<maxRank>::emplace(const tensorDimension (&array)[rank_]) {
        new(&data) Shape(array);
        currRank = rank_;
        return *this;
    }

    template<tensorRank maxRank>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) < maxRank)
    constexpr AnyShape<maxRank>& AnyShape<maxRank>::emplace(Dims ...dims) {
        new(&data) Shape(dims...);
        currRank = sizeof...(dims);
        return *this;
    }

    template<tensorRank maxRank>
    template<tensorRank newRank>
    requires(newRank <= maxRank)
    inline constexpr const Shape<newRank>* AnyShape<maxRank>::shape() const {
        if (currRank == newRank) {
            return reinterpret_cast<const Shape<newRank>*>(&data);
        } else {
            throw std::bad_any_cast();
        }
    }

    template<tensorRank maxRank>
    constexpr void AnyShape<maxRank>::reset() {
        currRank.reset();
    }

    template<tensorRank maxRank>
    template<tensorRank otherRank>
    requires(otherRank <= maxRank)
    constexpr AnyShape<maxRank>& AnyShape<maxRank>::operator=(const Shape<otherRank>& otherShape) {
        new(&data) Shape<otherRank>(otherShape);
        currRank = otherShape.rank();
        return *this;
    }

    template<tensorRank maxRank>
    template <tensorRank otherMaxRank>
    constexpr AnyShape<maxRank>& AnyShape<maxRank>::operator=(const AnyShape<otherMaxRank>& otherAnyShape) {
        operator=(otherAnyShape.shape());
        return *this;
    }

    template<tensorRank maxRank>
    template<tensorRank otherRank>
    requires(otherRank <= maxRank)
    constexpr bool AnyShape<maxRank>::operator==(const Shape<otherRank> &otherShape) const {
        return currRank.has_value()
               && currRank == otherRank
               && *shape<otherRank>() == otherShape;
    }

    template<tensorRank maxRank>
    template<tensorRank otherMaxRank>
    constexpr bool AnyShape<maxRank>::operator==(const AnyShape<otherMaxRank> &otherAnyShape) const {
        return currRank.has_value()
               && currRank == otherAnyShape.currRank
               && *shape<currRank>() == *otherAnyShape.shape();
    }

    template<tensorRank maxRank>
    constexpr tensorDimension AnyShape<maxRank>::operator[](tensorRank i) const {
        if (!currRank.has_value() || i >= currRank){
            throw std::runtime_error("Index out of range");
        }
        return (*shape<maxRank>())[i];
    }

    template<tensorRank maxRank>
    constexpr tensorRank AnyShape<maxRank>::rank() const {
        if (!currRank.has_value()){
            throw std::runtime_error("Cannot get rank of AnyShape in unassigned state");
        }
        return currRank.value();
    }

    template<tensorRank maxRank>
    constexpr tensorRank AnyShape<maxRank>::size() const {
        if (!currRank.has_value()){
            throw std::runtime_error("Cannot get size of AnyShape in unassigned state");
        }
        // Ideally, this would defer to Shape::size() instead of duplicating code.
        // Currently, I'm not sure how to cast it to only dynamically decidable shape
        auto is_positive = [](tensorDimension d) { return d > 0; };
        auto positives = reinterpret_cast<const Shape<maxRank>*>(&data)->dimensions
                         | std::views::take(currRank.value())
                         | std::views::filter(is_positive);
        return std::accumulate(positives.begin(), positives.end(), tensorDimension(1), std::multiplies());
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValidExplicit() const {
        // Ideally, this would defer to Shape::isValidExplicit() instead of duplicating code.
        // Currently, I'm not sure how to cast it to only dynamically decidable shape
        if (!currRank.has_value()) {
            return false;
        }
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = reinterpret_cast<const Shape<maxRank>*>(&data)->dimensions
                             | std::views::take(currRank.value())
                             | std::views::filter(is_non_positive);
        return non_positives.empty();
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValidImplicit() const {
        // Ideally, this would defer to Shape::isValidExplicit() instead of duplicating code.
        // Currently, I'm not sure how to cast it to only dynamically decidable shape
        if (!currRank.has_value()) {
            return false;
        }
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = reinterpret_cast<const Shape<maxRank>*>(&data)->dimensions
                             | std::views::take(currRank.value())
                             | std::views::filter(is_non_positive);
        return (std::ranges::distance(non_positives) == 1
                && (*non_positives.begin() == -1)
        );
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValid() const {
        return currRank.has_value() && (this->isValidExplicit() || this->isValidImplicit());
    }
}