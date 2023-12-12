//
// Created by Amy Fetzner on 12/10/2023.
//

#include <any>
#include "TensorII/private/headers/AnyShape_private.h"

namespace TensorII::Core {

    template<tensorRank maxRank>
    constexpr AnyShape<maxRank>::AnyShape()
    : data{}
    , currRank(std::nullopt)
    {}

    template<tensorRank maxRank>
    constexpr AnyShape<maxRank>::~AnyShape() = default;

    template<tensorRank maxRank>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) <= maxRank)
    constexpr AnyShape<maxRank>::AnyShape(const Dims ... dims)
    : data{0}
    {
        static_assert(std::is_trivially_copyable<Shape<maxRank>>::value);
        static_assert(std::is_trivially_copyable<tensorDimension[maxRank]>::value);
        tensorDimension* where = std::bit_cast<Shape<maxRank>>(data).dimensions.data();
        ([&where](tensorDimension dim){
            *where = dim;
            where = &where[1];
        }(dims), ...);
        currRank = sizeof...(dims);
    }

    template<tensorRank maxRank>
    template<tensorRank rank_>
    requires(rank_ <= maxRank)
    AnyShape<maxRank>& AnyShape<maxRank>::emplace(const tensorDimension (&array)[rank_]) {
        new(&data) Shape(array);
        currRank = rank_;
        return *this;
    }

    template<tensorRank maxRank>
    template<std::convertible_to<tensorDimension> ... Dims>
    requires(sizeof...(Dims) <= maxRank)
    AnyShape<maxRank>& AnyShape<maxRank>::emplace(const Dims ...dims) {
        new(&data) Shape(dims...);
        currRank = sizeof...(dims);
        return *this;
    }

    template<tensorRank maxRank>
    template<tensorRank newRank>
    requires(newRank <= maxRank)
    inline constexpr const Shape<newRank>* AnyShape<maxRank>::shape() const {
        if (currRank == newRank) {
            return &std::bit_cast<Shape<newRank>>(data);
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
//        // Ideally, this would defer to Shape::size() instead of duplicating code.
//        // Currently, I'm not sure how to cast it to only dynamically decidable shape
//        auto is_positive = [](tensorDimension d) { return d > 0; };
//        auto positives = data.dimensions
//                         | std::views::take(currRank.value())
//                         | std::views::filter(is_positive);
//        return std::accumulate(positives.begin(), positives.end(), tensorDimension(1), std::multiplies());
        return std::bit_cast<Shape<maxRank>>(data).size();
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValidExplicit() const {
        // Ideally, this would defer to Shape::isValidExplicit() instead of duplicating code.
        // Currently, I'm not sure how to cast it to only dynamically decidable shape
        if (!currRank.has_value()) {
            return false;
        }
        return std::bit_cast<Shape<maxRank>>(data).isValidExplicit();
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValidImplicit() const {
        // Ideally, this would defer to Shape::isValidExplicit() instead of duplicating code.
        // Currently, I'm not sure how to cast it to only dynamically decidable shape
        if (!currRank.has_value()) {
            return false;
        }
        return std::bit_cast<Shape<maxRank>>(data).isValidImplicit();
    }

    template<tensorRank maxRank>
    constexpr bool AnyShape<maxRank>::isValid() const {
        return std::bit_cast<Shape<maxRank>>(data).isValid();
    }
}