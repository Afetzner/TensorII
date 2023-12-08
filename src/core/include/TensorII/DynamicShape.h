//
// Created by Amy Fetzner on 12/6/2023.
//

#ifndef TENSOR_DYNAMICSHAPE_H
#define TENSOR_DYNAMICSHAPE_H

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

        constexpr Shape(const tensorDimension (&array)[rank]) { // NOLINT(google-explicit-constructor)
            std::copy_n(array, rank, dimensions.begin());
        }

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) == rank)
        constexpr Shape(const Dims& ... dims) : dimensions{dims...} {} // NOLINT(google-explicit-constructor)

        constexpr Shape(const Shape<rank>& other) {
            std::copy(other.dimensions.begin(), other.dimensions.end(), dimensions.begin());
        }

        Shape& operator=(const Shape<rank>& other) {
            if (this != &other){
                std::copy_n(other.dimensions.begin(), rank, dimensions.begin());
            }
            return *this;
        }

        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& other) {
            return (otherRank == rank) && std::equal(dimensions.begin(), dimensions.end(),
                                                     other.dimensions.begin(), other.dimensions.end());
        }

        [[nodiscard]]
        constexpr tensorSize size() const {
            // Product of all positive dimensions. size of an invalid shape is undefined
            auto is_positive = [](tensorDimension d) { return d > 0; };
            auto positives = dimensions | std::views::filter(is_positive);
            return std::accumulate(positives.begin(), positives.end(), tensorDimension(1), std::multiplies());
        }

        [[nodiscard]]
        constexpr bool isValidExplicit() const {
            // Valid explicit if there are no non-positive dimensions
            auto is_non_positive = [](tensorDimension d) { return d <= 0; };
            auto non_positives
                    = std::views::all(dimensions)
                    | std::views::filter(is_non_positive);
            return non_positives.empty();
        }

        [[nodiscard]]
        constexpr bool isValidImplicit() const {
            // Valid explicit if there is only one non-positive dimension, and it is -1
            auto is_non_positive = [](tensorDimension d) { return d <= 0; };
            auto non_positives
                    = std::views::all(dimensions)
                    | std::views::filter(is_non_positive);
            return (std::ranges::distance(non_positives) == 1
                    && (*non_positives.begin() == -1)
                    );
        }

        [[nodiscard]]
        constexpr bool isValid() const {
            return this->isValidExplicit() || this->isValidImplicit();
        }
    };

    Shape() -> Shape<0>;
    Shape(tensorDimension) -> Shape<1>;

    template<std::convertible_to<tensorDimension> ... Dims>
    Shape(Dims ... dims) -> Shape<sizeof...(dims)>;

    template<tensorRank N>
    Shape(const tensorDimension (&array)[N]) -> Shape<N>;

    template<tensorRank explicitRank, tensorRank implicitRank>
    constexpr Shape<implicitRank> DeduceShape(const Shape<explicitRank>& explicitShape, const Shape<implicitRank>& implicitShape) {
        if (!explicitShape.isValidExplicit()){
            throw std::runtime_error("Arg 'explicitShape' is not a valid explicit shape");
        }
        if (!implicitShape.isValidImplicit()){
            throw std::runtime_error("Arg 'implicitShape' is not a valid implicit shape");
        }

        const tensorDimension deducedDim = explicitShape.size() / implicitShape.size();
        const tensorDimension remainder  = explicitShape.size() % implicitShape.size();
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

#endif //TENSOR_DYNAMICSHAPE_H
