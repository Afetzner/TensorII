//
// Created by Amy Fetzner on 12/8/2023.
//

#include "TensorII/private/Shape_private.h"

namespace TensorII::Core {

    template<tensorRank rank_>
    constexpr Shape<rank_>::Shape(const tensorDimension (&array)[rank]) {
        std::copy_n(array, rank, dimensions.begin());
    }

    template<tensorRank rank_>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) == rank_)
    constexpr Shape<rank_>::Shape(const Dims &... dims) : dimensions{dims...} {}

    template<tensorRank rank_>
    constexpr Shape<rank_>::Shape(const Shape<rank_> &other) {
        std::copy(other.dimensions.begin(), other.dimensions.end(), dimensions.begin());
    }

    template<tensorRank rank_>
    constexpr Shape<rank_> &Shape<rank_>::operator=(const Shape<rank> &other) {
        if (this != &other) {
            std::copy_n(other.dimensions.begin(), rank, dimensions.begin());
        }
        return *this;
    }

    template<tensorRank rank_>
    template<tensorRank otherRank>
    constexpr bool Shape<rank_>::operator==(const Shape<otherRank> &other) {
        return (otherRank == rank) && std::equal(dimensions.begin(), dimensions.end(),
                                                 other.dimensions.begin(), other.dimensions.end());
    }

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
        auto non_positives
                = std::views::all(dimensions)
                  | std::views::filter(is_non_positive);
        return non_positives.empty();
    }

    template<tensorRank rank_>
    constexpr bool Shape<rank_>::isValidImplicit() const {
        // Valid explicit if there is only one non-positive dimension, and it is -1
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives
                = std::views::all(dimensions)
                  | std::views::filter(is_non_positive);
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
    DeduceShape(const Shape<explicitRank> &explicitShape, const Shape<implicitRank> &implicitShape) {
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
}