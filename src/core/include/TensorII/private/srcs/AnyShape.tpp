//
// Created by Amy Fetzner on 12/10/2023.
//
#ifndef TENSOR_ANYSHAPE_TPP
#define TENSOR_ANYSHAPE_TPP

#include <any>
#include <algorithm>
#include "TensorII/private/headers/AnyShape_private.h"

namespace TensorII::Core {

    template<tensorRank maxRank>
    constexpr AnyShape<maxRank>::~AnyShape() = default;

    template<tensorRank maxRank>
    constexpr AnyShape<maxRank>::AnyShape()
    : dimensions{}
    , currRank(0)
    {}

    template<tensorRank maxRank>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) <= maxRank)
    constexpr AnyShape<maxRank>::AnyShape(const Dims ... dims)
    : dimensions{}
    {
        auto where = dimensions.begin();
        ([&where](tensorDimension dim){
            *where = dim;
            where++;
        }(dims), ...);
        currRank = sizeof...(dims);
    }

    template<tensorRank maxRank>
    template<tensorRank rank_>
    requires(rank_ <= maxRank)
    constexpr AnyShape<maxRank>::AnyShape(const Shape<rank_>& shape)
    : dimensions{}
    {
        std::ranges::copy_n(shape.dimensions.begin(), rank_, dimensions.begin());
        currRank = rank_;
    }

    template<tensorRank maxRank>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    constexpr AnyShape<maxRank>::AnyShape(from_range_t, Range&& range)
    {
        tensorRank n_new = std::ranges::size(range);
        if (n_new > maxRank) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        // Copy new dimensions
        std::ranges::copy_n(range.begin(), n_new, dimensions.begin());
        currRank = n_new;
    }

    template<tensorRank maxRank>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    AnyShape<maxRank>& AnyShape<maxRank>::emplace(from_range_t, Range&& range)
    {
       tensorRank n_new = std::ranges::size(range);
        if (n_new > maxRank) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        // Add new dimensions
        std::ranges::copy_n(range.begin(), n_new, dimensions.begin());
        currRank = n_new;
        return *this;
    }

    template<tensorRank maxRank>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) <= maxRank)
    AnyShape<maxRank>& AnyShape<maxRank>::emplace(const Dims ...dims) {
        auto where = dimensions.begin();
        ([&where](tensorDimension dim){
            *where = dim;
            where++;
        }(dims), ...);
        currRank = sizeof...(dims);
        return *this;
    }

    template<tensorRank maxRank>
    template<tensorRank newRank>
    requires(newRank <= maxRank)
    constexpr Shape<newRank> AnyShape<maxRank>::shape() const {
        if (currRank != newRank) {
            throw std::bad_any_cast();
        }
        Shape<newRank> newShape;
        std::ranges::copy_n(dimensions.begin(), newRank, newShape.dimensions.begin());
        return newShape;
    }

    //region Constexpr augment & demote
    template<tensorRank maxRank>
    template<std::convertible_to<tensorDimension>... Dims>
    constexpr AnyShape<maxRank> AnyShape<maxRank>::augmented(const Dims... dims) const {
        tensorRank n_old = currRank;
        tensorRank n_new = sizeof...(Dims);
        if (n_new > maxRank - n_old) {
            throw std::length_error("Length of dimensions exceeds capacity of shape");
        }
        AnyShape<maxRank> newShape;
        // Copy old dimensions
        std::ranges::copy_n(dimensions.begin(), n_old, newShape.dimensions.begin());
        // Add new dimensions
        auto where = newShape.dimensions.begin() + n_old;
        ([&where](tensorDimension dim){
            *where = dim;
            where++;
        }(dims), ...);
        newShape.currRank = n_old + n_new;
        return newShape;
    }

    template<tensorRank maxRank>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    constexpr AnyShape<maxRank> AnyShape<maxRank>::augmented(from_range_t, Range&& range) const
    {
        tensorRank n_old = currRank;
        tensorRank n_new = std::ranges::size(range);
        if (n_new > maxRank - n_old) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        AnyShape<maxRank> newShape;
        // Copy old dimensions
        std::ranges::copy_n(dimensions.begin(), n_old, newShape.dimensions.begin());
        // Add new dimensions
        std::ranges::copy_n(range.begin(), n_new, newShape.dimensions.begin() + n_old);
        newShape.currRank = n_old + n_new;
        return newShape;
    }

    template<tensorRank maxRank>
    constexpr AnyShape<maxRank> AnyShape<maxRank>::demoted(tensorRank newRank) const {
        tensorRank n_old = currRank;
        if (newRank > n_old) {
            throw std::length_error("New shape rank exceeds old shape's rank");
        }
        AnyShape<maxRank> newShape;
        // Copy old dimensions
        std::ranges::copy_n(dimensions.begin(), newRank, newShape.dimensions.begin());
        newShape.currRank = newRank;
        return newShape;
    }
    //endregion Constexpr augment & demote

    //region Non-constexpr augment & demote
    template<tensorRank maxRank>
    template<std::convertible_to<tensorDimension>... Dims>
    void AnyShape<maxRank>::augment(const Dims... dims)
    {
        tensorRank n_old = currRank;
        tensorRank n_new = sizeof...(Dims);
        if (n_new == 0) { return; }
        if (n_new > maxRank - n_old) {
            throw std::length_error("Length of dimensions exceeds capacity of shape");
        }
        // Add new dimensions
        tensorDimension* where = &dimensions[n_old];
        ([&where](tensorDimension dim){
            *where = dim;
            where = &where[1];
        }(dims), ...);
        currRank = n_old + n_new;
    }

    template<tensorRank maxRank>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    void AnyShape<maxRank>::augment(from_range_t, Range&& range) {
        tensorRank n_old = currRank;
        tensorRank n_new = std::ranges::size(range);
        if (n_new == 0) { return; }
        if (n_new > maxRank - n_old) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        // Add new dimensions
        std::ranges::copy_n(range.begin(), n_new, dimensions.begin() + n_old);
        currRank = n_old + n_new;
    }

    template<tensorRank maxRank>
    void AnyShape<maxRank>::demote(tensorRank newRank) {
        tensorRank n_old = currRank;
        if (newRank > n_old) {
            throw std::length_error("New shape rank exceeds old shape's rank");
        }
        currRank = newRank;
    }
    //endregion Non-constexpr augment & demote

    template<tensorRank maxRank>
    constexpr void AnyShape<maxRank>::reset() {
        currRank = 0;
    }

    template<tensorRank maxRank>
    template<tensorRank otherRank>
    requires(otherRank <= maxRank)
    AnyShape<maxRank>& AnyShape<maxRank>::operator=(const Shape<otherRank>& otherShape) {
        std::ranges::copy_n(otherShape.dimensions.begin(), otherRank, dimensions.begin());
        currRank = otherRank;
        return *this;
    }

    template<tensorRank maxRank>
    template <tensorRank otherMaxRank>
    AnyShape<maxRank>& AnyShape<maxRank>::operator=(const AnyShape<otherMaxRank>& otherAnyShape) {
        tensorRank newRank = otherAnyShape.rank();
        if (newRank > maxRank) {
            throw std::length_error("Rank of assignee exceeds capacity of shape");
        }
        std::ranges::copy_n(otherAnyShape.dimensions.begin(), newRank, dimensions.begin());
        currRank = newRank;
        return *this;
    }

    template<tensorRank maxRank>
    template<tensorRank otherRank>
    constexpr bool AnyShape<maxRank>::operator==(const Shape<otherRank> &otherShape) const {
        if (currRank != otherRank) {
            return false;
        }
        if (otherRank == 0) {
            return true;
        }
        tensorDimension how_many = currRank;
        auto us = dimensions | std::views::take(how_many);
        auto them = otherShape.dimensions | std::views::take(how_many);
        return std::ranges::equal(us, them);
    }

    template<tensorRank maxRank>
    template<tensorRank otherMaxRank>
    constexpr bool AnyShape<maxRank>::operator==(const AnyShape<otherMaxRank> &otherAnyShape) const {
        if (this == &otherAnyShape){
            return true;
        }
        if (currRank != otherAnyShape.rank()) {
            return false;
        }

        tensorDimension how_many = currRank;
        auto us = dimensions | std::views::take(how_many);
        auto them = otherAnyShape.dimensions | std::views::take(how_many);
        return std::ranges::equal(us, them);
    }

    template<tensorRank maxRank>
    constexpr tensorDimension AnyShape<maxRank>::operator[](tensorRank i) const {
        if (i >= currRank){
            throw std::range_error("Index out of range");
        }
        return (*shape<maxRank>())[i];
    }

    template<tensorRank maxRank>
    constexpr tensorRank AnyShape<maxRank>::rank() const {
        return currRank;
    }

    template<tensorRank maxRank>
    constexpr tensorRank AnyShape<maxRank>::size() const {
        auto is_positive = [](tensorDimension d) { return d > 0; };
        auto positives = dimensions
                         | std::views::take(currRank)
                         | std::views::filter(is_positive);
        return std::accumulate(positives.begin(), positives.end(), tensorDimension(1), std::multiplies());
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValidExplicit() const {
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = dimensions
                             | std::views::take(currRank)
                             | std::views::filter(is_non_positive);
        return non_positives.empty();
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValidImplicit() const {
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = dimensions
                             | std::views::take(currRank)
                             | std::views::filter(is_non_positive);
        return (std::ranges::distance(non_positives) == 1
                && (*non_positives.begin() == -1)
        );
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValid() const {
        return isValidExplicit() || isValidImplicit();
    }
}

#endif //TENSOR_ANYSHAPE_TPP
