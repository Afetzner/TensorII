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

#include "TensorII/private/ConceptUtil.h"
#include "TensorII/Types.h"

namespace TensorII::Core {

    template<tensorRank> struct Shape;

    class ShapeIterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = tensorDimension;
        using difference_type   = std::ptrdiff_t;
        using pointer           = tensorDimension *;
        using reference         = tensorDimension &;

        explicit constexpr ShapeIterator(tensorDimension* ptr = nullptr) : m_ptr(ptr) {}
        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }
        ShapeIterator& operator++() { m_ptr++; return *this; }
        ShapeIterator operator++(int) { ShapeIterator tmp = *this; ++(*this); return tmp; } // NOLINT(cert-dcl21-cpp)
        friend bool operator== (const ShapeIterator& a, const ShapeIterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!= (const ShapeIterator& a, const ShapeIterator& b) { return a.m_ptr != b.m_ptr; };
    private:
        pointer m_ptr;
    };

    template <tensorRank rank_>
    struct Shape {
        std::array<tensorDimension, std::max(rank_, 1U)> dimensions;

        constexpr Shape();
        constexpr ~Shape() = default;
        constexpr Shape(const Shape<rank_>& other) = default;
        constexpr Shape<rank_>& operator=(const Shape<rank_>& other) = default;

        constexpr Shape(const tensorDimension (&array)[rank_]); // NOLINT(google-explicit-constructor)

        template<Util::ContainerCompatibleRange<tensorDimension> Range>
        constexpr explicit Shape(from_range_t, Range&& range);

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) == rank_)
        explicit constexpr Shape(const Dims& ... dims);

        template <std::convertible_to<tensorDimension> ... Dims>
        constexpr Shape<rank_ + sizeof...(Dims)> augmented(const Dims ... dims) const;

        template<tensorRank newRank, Util::ContainerCompatibleRange<tensorDimension> Range>
        constexpr Shape<newRank> augmented(from_range_t, Range&& augmentDimensions) const;

        template <tensorRank newRank>
        requires(newRank < rank_ && newRank != 0)
        constexpr Shape<newRank> demoted() const;

        template <tensorRank newRank>
        requires(newRank == 0)
        constexpr Shape<newRank> demoted() const;

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

        [[nodiscard]]
        constexpr ShapeIterator begin() const { return ShapeIterator{dimensions.data()}; }

        [[nodiscard]]
        constexpr ShapeIterator end() const {
            if (rank_ != 0) { return ShapeIterator{dimensions.data() + dimensions.size()}; }
            return ShapeIterator{dimensions.data()};
        }
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
