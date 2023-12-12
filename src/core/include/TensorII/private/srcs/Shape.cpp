//
// Created by Amy Fetzner on 12/8/2023.
//

#include "TensorII/private/headers/Shape_private.h"
#include <ranges>

namespace TensorII::Core {

    template<tensorRank rank_>
    constexpr Shape<rank_>::Shape() : dimensions {0} {}

    template<tensorRank rank_>
    constexpr Shape<rank_>::Shape(const tensorDimension (&array)[rank_]){
        std::copy_n(array, rank_, dimensions.begin());
    }

    template<tensorRank rank_>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) == rank_)
    constexpr Shape<rank_>::Shape(const Dims &... dims) : dimensions{dims...} {}

    template<tensorRank rank_>
    template<std::convertible_to<tensorDimension>... Dims>
    constexpr Shape<rank_ + sizeof...(Dims)> Shape<rank_>::augment(const Dims... dims) const {
        Shape<rank_ + sizeof...(Dims)> newShape{};
        std::ranges::copy(dimensions.begin(), dimensions.end(),
                          newShape.dimensions.begin());
        tensorDimension* where = &newShape.dimensions[rank_];
        ([&where](tensorDimension dim){
            *where = dim;
            where = &where[1];
        }(dims), ...);
        return newShape;
    }

    template<tensorRank rank_>
    template<tensorRank rankDiff>
    constexpr Shape<rank_ + rankDiff> Shape<rank_>::augment(const tensorDimension (&array)[rankDiff]) const {
        Shape<rank_ + rankDiff> newShape{};
        std::ranges::copy(dimensions.begin(), dimensions.end(),
                          newShape.dimensions.begin());
        std::ranges::copy(array, newShape.dimensions.begin() + rank_);
        return newShape;
    }

    template<tensorRank rank_>
    template<tensorRank newRank>
    requires(newRank < rank_ && newRank != 0)
    constexpr Shape<newRank> Shape<rank_>::demote() const {
        Shape<newRank> newShape{};
        std::ranges::copy(dimensions.begin(), dimensions.begin() + newRank,
                          newShape.dimensions.begin());
        return newShape;
    }

    template<tensorRank rank_>
    template<tensorRank newRank>
    requires(newRank == 0)
    constexpr Shape<newRank> Shape<rank_>::demote() const {
        return Shape<newRank>{};
    }

    template<tensorRank rank_>
    template<tensorRank otherRank>
    constexpr bool Shape<rank_>::operator==(const Shape<otherRank> &other) const {
        return (otherRank == rank_) && std::equal(dimensions.begin(), dimensions.end(),
                                                 other.dimensions.begin(), other.dimensions.end());
    }

    template<tensorRank rank_>
    inline constexpr tensorDimension Shape<rank_>::operator[](tensorRank i) const {
        return dimensions[i];
    }

    template<tensorRank rank_>
    constexpr tensorRank Shape<rank_>::rank() const { return rank_; }

    template<tensorRank rank_>
    constexpr tensorSize Shape<rank_>::size() const {
        // Product of all positive dimensions. size of an invalid shape is undefined
        auto is_positive = [](tensorDimension d) { return d > 0; };
        auto positives = dimensions | std::views::filter(is_positive);
        return std::accumulate(positives.begin(), positives.end(), tensorDimension(1), std::multiplies());
    }

    template<tensorRank rank_>
    constexpr bool Shape<rank_>::isValidExplicit() const {
        // Valid explicit if there are no non-positive dimensions
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = dimensions | std::views::filter(is_non_positive);
        return non_positives.empty();
    }

    template<tensorRank rank_>
    constexpr bool Shape<rank_>::isValidImplicit() const {
        // Valid explicit if there is only one non-positive dimension, and it is -1
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = dimensions | std::views::filter(is_non_positive);
        return (std::ranges::distance(non_positives) == 1
                && (*non_positives.begin() == -1)
        );
    }

    template<tensorRank rank_>
    constexpr bool Shape<rank_>::isValid() const {
        return this->isValidExplicit() || this->isValidImplicit();
    }

    template<tensorRank explicitRank, tensorRank implicitRank>
    constexpr Shape<implicitRank>
    deduceShape(const Shape<explicitRank> &explicitShape, const Shape<implicitRank> &implicitShape) {
        if (!explicitShape.isValidExplicit()) {
            throw std::runtime_error("Arg 'explicitShape' is not a valid explicit shape");
        }
        if (!implicitShape.isValidImplicit()) {
            throw std::runtime_error("Arg 'implicitShape' is not a valid implicit shape");
        }

        const tensorDimension deducedDim = explicitShape.size() / implicitShape.size();
        const tensorDimension remainder = explicitShape.size() % implicitShape.size();
        if (remainder != 0) {
            throw std::runtime_error("Cannot deduce shape using incompatible shapes");
        }
        Shape newShape = Shape<implicitRank>(implicitShape);
        auto negative = std::find_if(newShape.dimensions.begin(), newShape.dimensions.end(),
                                     [](tensorDimension d) { return d == -1; });
        *negative = deducedDim;
        return newShape;
    }

    template<std::convertible_to<tensorDimension>... Dims>
    inline constexpr Shape<sizeof...(Dims)> Shape<0>::augment(const Dims... dims) const {
        return Shape<sizeof...(Dims)>{dims ... };
    }

    template<tensorRank rankDiff>
    inline constexpr Shape<rankDiff> Shape<0>::augment(const tensorDimension (&array)[rankDiff]) const {
        return Shape<rankDiff>(array);
    }
}