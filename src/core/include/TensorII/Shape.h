//
// Created by Amy Fetzner on 12/6/2023.
//

#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

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

    template <tensorRank rank_>
    struct Shape {
        std::array<tensorDimension, std::max(rank_, (tensorRank)1)> dimensions;

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
        constexpr tensorSize size() const { return rank_; };

        [[nodiscard]]
        constexpr tensorSize n_elems() const; // Can't be called size, or it will use this for the range funcs

        [[nodiscard]]
        constexpr bool isValidExplicit() const;

        [[nodiscard]]
        constexpr bool isValidImplicit() const;

        [[nodiscard]]
        constexpr bool isValid() const;

        class Iterator {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type        = tensorDimension;
            using difference_type   = std::ptrdiff_t;
            using pointer           = const tensorDimension *;
            using reference         = const tensorDimension &;

            explicit constexpr Iterator(const tensorDimension* ptr = nullptr) : m_ptr(ptr) {}
            constexpr reference operator*() const { return *m_ptr; }
            constexpr pointer operator->() { return m_ptr; }
            constexpr Iterator& operator++() { m_ptr++; return *this; }
            constexpr Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; } // NOLINT(cert-dcl21-cpp)
            constexpr friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
            constexpr friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };
        private:
            pointer m_ptr;
        };

        [[nodiscard]]
        constexpr Iterator begin() const { return Iterator{dimensions.data()}; }

        [[nodiscard]]
        constexpr Iterator end() const {
            if (rank_ != 0) { return Iterator{dimensions.data() + dimensions.size()}; }
            return Iterator{dimensions.data()};
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

#endif //TENSOR_SHAPE_H

#include "TensorII/private/templates/Shape.tpp" // NOLINT(bugprone-suspicious-include)
