//
// Created by Amy Fetzner on 12/10/2023.
//
#ifndef TENSOR_ANYSHAPE_TPP
#define TENSOR_ANYSHAPE_TPP

#include <any>
#include <algorithm>
#include "TensorII/AnyShape.h"

namespace TensorII::Core {

    template<tensorRank maxRank_>
    constexpr AnyShape<maxRank_>::~AnyShape() = default;

    template<tensorRank maxRank_>
    constexpr AnyShape<maxRank_>::AnyShape()
    : dimensions{}
    , currRank(0)
    {}

    template<tensorRank maxRank_>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) <= maxRank_)
    constexpr AnyShape<maxRank_>::AnyShape(const Dims ... dims)
    : dimensions{}
    {
        auto where = dimensions.begin();
        ([&where](tensorDimension dim){
            *where = dim;
            where++;
        }(dims), ...);
        currRank = sizeof...(dims);
    }

    template<tensorRank maxRank_>
    template<tensorRank rank_>
    requires(rank_ <= maxRank_)
    constexpr AnyShape<maxRank_>::AnyShape(const Shape<rank_>& shape)
    : dimensions{}
    {
        std::ranges::copy_n(shape.dimensions.begin(), rank_, dimensions.begin());
        currRank = rank_;
    }

    template<tensorRank maxRank_>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    constexpr AnyShape<maxRank_>::AnyShape(from_range_t, Range&& range)
    {
        tensorRank n_new = std::ranges::size(range);
        if (n_new > maxRank_) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        // Copy new dimensions
        std::ranges::copy_n(range.begin(), n_new, dimensions.begin());
        currRank = n_new;
    }

    template<tensorRank maxRank_>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    AnyShape<maxRank_>& AnyShape<maxRank_>::emplace(from_range_t, Range&& range)
    {
       tensorRank n_new = std::ranges::size(range);
        if (n_new > maxRank_) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        // Add new dimensions
        std::ranges::copy_n(range.begin(), n_new, dimensions.begin());
        currRank = n_new;
        return *this;
    }

    template<tensorRank maxRank_>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) <= maxRank_)
    AnyShape<maxRank_>& AnyShape<maxRank_>::emplace(const Dims ...dims) {
        auto where = dimensions.begin();
        ([&where](tensorDimension dim){
            *where = dim;
            where++;
        }(dims), ...);
        currRank = sizeof...(dims);
        return *this;
    }

    template<tensorRank maxRank_>
    template<tensorRank newRank>
    requires(newRank <= maxRank_)
    constexpr Shape<newRank> AnyShape<maxRank_>::shape() const {
        if (currRank != newRank) {
            throw std::bad_any_cast();
        }
        Shape<newRank> newShape;
        std::ranges::copy_n(dimensions.begin(), newRank, newShape.dimensions.begin());
        return newShape;
    }

    //region Constexpr augment & demote
    template<tensorRank maxRank_>
    template<std::convertible_to<tensorDimension>... Dims>
    constexpr AnyShape<maxRank_> AnyShape<maxRank_>::augmented(const Dims... dims) const {
        tensorRank n_old = currRank;
        tensorRank n_new = sizeof...(Dims);
        if (n_new > maxRank_ - n_old) {
            throw std::length_error("Length of dimensions exceeds capacity of shape");
        }
        AnyShape<maxRank_> newShape;
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

    template<tensorRank maxRank_>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    constexpr AnyShape<maxRank_> AnyShape<maxRank_>::augmented(from_range_t, Range&& range) const
    {
        tensorRank n_old = currRank;
        tensorRank n_new = std::ranges::size(range);
        if (n_new > maxRank_ - n_old) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        AnyShape<maxRank_> newShape;
        // Copy old dimensions
        std::ranges::copy_n(dimensions.begin(), n_old, newShape.dimensions.begin());
        // Add new dimensions
        std::ranges::copy_n(range.begin(), n_new, newShape.dimensions.begin() + n_old);
        newShape.currRank = n_old + n_new;
        return newShape;
    }

    template<tensorRank maxRank_>
    constexpr AnyShape<maxRank_> AnyShape<maxRank_>::demoted(tensorRank newRank) const {
        tensorRank n_old = currRank;
        if (newRank > n_old) {
            throw std::length_error("New shape rank exceeds old shape's rank");
        }
        AnyShape<maxRank_> newShape;
        // Copy old dimensions
        std::ranges::copy_n(dimensions.begin(), newRank, newShape.dimensions.begin());
        newShape.currRank = newRank;
        return newShape;
    }
    //endregion Constexpr augment & demote

    //region Non-constexpr augment & demote
    template<tensorRank maxRank_>
    template<std::convertible_to<tensorDimension>... Dims>
    void AnyShape<maxRank_>::augment(const Dims... dims)
    {
        tensorRank n_old = currRank;
        tensorRank n_new = sizeof...(Dims);
        if (n_new == 0) { return; }
        if (n_new > maxRank_ - n_old) {
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

    template<tensorRank maxRank_>
    template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
    void AnyShape<maxRank_>::augment(from_range_t, Range&& range) {
        tensorRank n_old = currRank;
        tensorRank n_new = std::ranges::size(range);
        if (n_new == 0) { return; }
        if (n_new > maxRank_ - n_old) {
            throw std::length_error("Length of range exceeds capacity of shape");
        }
        // Add new dimensions
        std::ranges::copy_n(range.begin(), n_new, dimensions.begin() + n_old);
        currRank = n_old + n_new;
    }

    template<tensorRank maxRank_>
    void AnyShape<maxRank_>::demote(tensorRank newRank) {
        tensorRank n_old = currRank;
        if (newRank > n_old) {
            throw std::length_error("New shape rank exceeds old shape's rank");
        }
        currRank = newRank;
    }
    //endregion Non-constexpr augment & demote

    template<tensorRank maxRank_>
    constexpr void AnyShape<maxRank_>::reset() {
        currRank = 0;
    }

    template<tensorRank maxRank_>
    template<tensorRank otherRank>
    requires(otherRank <= maxRank_)
    AnyShape<maxRank_>& AnyShape<maxRank_>::operator=(const Shape<otherRank>& otherShape) {
        std::ranges::copy_n(otherShape.dimensions.begin(), otherRank, dimensions.begin());
        currRank = otherRank;
        return *this;
    }

    template<tensorRank maxRank_>
    template <tensorRank otherMaxRank>
    AnyShape<maxRank_>& AnyShape<maxRank_>::operator=(const AnyShape<otherMaxRank>& otherAnyShape) {
        tensorRank newRank = otherAnyShape.rank();
        if (newRank > maxRank_) {
            throw std::length_error("Rank of assignee exceeds capacity of shape");
        }
        std::ranges::copy_n(otherAnyShape.dimensions.begin(), newRank, dimensions.begin());
        currRank = newRank;
        return *this;
    }

    template<tensorRank maxRank_>
    template<tensorRank otherRank>
    constexpr bool AnyShape<maxRank_>::operator==(const Shape<otherRank> &otherShape) const {
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

    template<tensorRank maxRank_>
    template<tensorRank otherMaxRank>
    constexpr bool AnyShape<maxRank_>::operator==(const AnyShape<otherMaxRank> &otherAnyShape) const {
        if constexpr (maxRank_ == otherMaxRank){
            if (this == &otherAnyShape) {
                return true;
            }
        }
        if (currRank != otherAnyShape.rank()) {
            return false;
        }

        tensorDimension how_many = currRank;
        auto us = dimensions | std::views::take(how_many);
        auto them = otherAnyShape.dimensions | std::views::take(how_many);
        return std::ranges::equal(us, them);
    }

    template<tensorRank maxRank_>
    constexpr tensorDimension AnyShape<maxRank_>::operator[](tensorRank axis) const {
        if (axis >= currRank){
            throw std::out_of_range("Index out of range");
        }
        return dimensions[axis];
    }

    template<tensorRank maxRank_>
    inline constexpr tensorRank AnyShape<maxRank_>::maxRank() const {
        return maxRank_;
    }

    template<tensorRank maxRank_>
    inline constexpr tensorRank AnyShape<maxRank_>::rank() const {
        return currRank;
    }

    template<tensorRank maxRank_>
    inline constexpr tensorRank AnyShape<maxRank_>::size() const {
        return currRank;
    }

    template<tensorRank maxRank_>
    constexpr tensorRank AnyShape<maxRank_>::n_elems() const {
        auto is_positive = [](tensorDimension d) { return d > 0; };
        auto positives = dimensions
                         | std::views::take(currRank)
                         | std::views::filter(is_positive);
        return std::accumulate(positives.begin(), positives.end(), tensorDimension(1), std::multiplies());
    }

    template<tensorRank maxRank_>
    constexpr bool AnyShape<maxRank_>::isValidExplicit() const {
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = dimensions
                             | std::views::take(currRank)
                             | std::views::filter(is_non_positive);
        return non_positives.empty();
    }

    template<tensorRank maxRank_>
    constexpr bool AnyShape<maxRank_>::isValidImplicit() const {
        auto is_non_positive = [](tensorDimension d) { return d <= 0; };
        auto non_positives = dimensions
                             | std::views::take(currRank)
                             | std::views::filter(is_non_positive);
        return (std::ranges::distance(non_positives) == 1
                && (*non_positives.begin() == -1)
        );
    }

    template<tensorRank maxRank_>
    constexpr bool AnyShape<maxRank_>::isValid() const {
        return isValidExplicit() || isValidImplicit();
    }

    template<tensorRank maxRank_>
    constexpr AnyShape<maxRank_>::ConstIterator AnyShape<maxRank_>::end() const {
        if (currRank != 0) {
            return ConstIterator(dimensions.data() + currRank);
        }
        return ConstIterator {dimensions.data()};
    }

    template<tensorRank maxRank_>
    constexpr AnyShape<maxRank_>::Iterator AnyShape<maxRank_>::end() {
        if (currRank != 0) {
            return Iterator(dimensions.data() + currRank);
        }
        return Iterator {dimensions.data()};
    }


    template<tensorRank maxRank_>
    constexpr AnyShape<maxRank_>::ConstIterator AnyShape<maxRank_>::begin() const {
        return ConstIterator{dimensions.data()};
    }

    template<tensorRank maxRank_>
    constexpr AnyShape<maxRank_>::Iterator AnyShape<maxRank_>::begin() {
        return Iterator{dimensions.data()};
    }

}

#endif //TENSOR_ANYSHAPE_TPP
