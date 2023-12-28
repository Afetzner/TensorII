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

    template <tensorRank rank_>
    struct Shape {
        std::array<tensorDimension, std::max(rank_, (tensorRank)1)> dimensions;

        constexpr Shape();
        constexpr ~Shape() = default;
        constexpr Shape(const Shape<rank_>& other) = default;
        constexpr Shape<rank_>& operator=(const Shape<rank_>& other) = default;

        /// Create a Shape from a c-style array, copying its values
        constexpr Shape(const tensorDimension (&array)[rank_]);  // NOLINT(google-explicit-constructor)

        /// Create a Shape from a range of tensorDimensions.
        /// If the range is sized, it will check that it's length matches. On length mismatch, throws std::length_error
        template<Util::ContainerCompatibleRange<tensorDimension> Range>
        constexpr explicit Shape(from_range_t, Range&& range);

        /// Create a shape from a variadic number of tensorDimensions
        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) == rank_)
        constexpr Shape(const Dims& ... dims);  // NOLINT(google-explicit-constructor)

        /// Creates a new shape that is the result of appending the new dimensions 'dims' to the existing ones
        template <std::convertible_to<tensorDimension> ... Dims>
        constexpr Shape<rank_ + sizeof...(Dims)> augmented(const Dims ... dims) const;

        /// Creates a new shape that is the result of appending the range the existing dimensions
        /// Requires that the user inform how long the new range will be
        /// If the range is sized, it will check that its length matches. On length mismatch, throws std::length_error
        template<tensorRank newRank, Util::ContainerCompatibleRange<tensorDimension> Range>
        constexpr Shape<newRank> augmented(from_range_t, Range&& augmentDimensions) const;

        /// Creates a new shape that is the 'newRank' dimensions of the existing ones
        template <tensorRank newRank>
        requires(newRank < rank_)
        constexpr Shape<newRank> demoted() const;

        /// Creates a new shape that is the existing dimensions, except the one at 'axis' is replace with 'newDimension'
        /// If axis is not a valid axis of the shape, throws std::out_of_bounds
        constexpr Shape<rank_> replace(tensorRank axis, tensorDimension newDimension) const;

        /// Two shapes are equal if they are the same length and every dimension is equal piece-wise
        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& other) const;

        /// Returns an r-value of the dimension at the given axis
        /// On indexing on invalid axis, throws std::out_of_range
        constexpr tensorDimension operator[](tensorRank axis) const;

        /// Returns the rank - the number of dimensions - of the shape
        [[nodiscard]]
        constexpr tensorRank rank() const noexcept { return rank_; };

        /// Returns the number of dimensions, to be used for the ranges interface.
        /// Not to be confused with the number of elements in the shape: n_elems
        [[nodiscard]] [[maybe_unused]]
        constexpr tensorSize size() const noexcept { return rank_; };

        /// Returns the number of elements in an array of this shape. I.e. Shape{2, 5, 7} has a 2*5*7=70 elements
        /// Not to be confused with the size or rank
        [[nodiscard]]
        constexpr tensorSize n_elems() const; // Can't be called size, or it will use this for the range funcs

        /// Returns if the shape is a valid explicit shape
        /// A shape is a valid explicit shape iff it has only positive dimensions
        [[nodiscard]]
        constexpr bool isValidExplicit() const;

        /// Returns if the shape is a valid implicit shape
        /// A shape is a valid implicit shape iff it has only one non-positive dimension and it is -1
        [[nodiscard]]
        constexpr bool isValidImplicit() const;

        /// Returns if the shape is a valid explicit shape or a valid implicit shape
        [[nodiscard]]
        constexpr bool isValid() const;

        /// An constant, random-access iterator to a shape. It allows reads but not writes
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

        /// Returns an iterator to the beginning of the shape
        [[nodiscard]]
        constexpr Iterator begin() const;

        /// Returns an iterator to the end of the shape. For a 0-rank shape, it returns the beginning
        [[nodiscard]]
        constexpr Iterator end() const;
    };

    Shape() -> Shape<0>;

    Shape(tensorDimension) -> Shape<1>;

    // Allows implicitly turing a pack of dimensions into a shape
    template<std::convertible_to<tensorDimension> ... Dims>
    Shape(Dims ... dims) -> Shape<sizeof...(dims)>;

    // Allows implicitly turning a c-style array into a shape
    template<tensorRank N>
    Shape(const tensorDimension (&array)[N]) -> Shape<N>;

    /// Given an explicit shape, and an implicit shape, deduces the unknown dimensions such that the two have the same number of elements
    /// I.e. deduceShape({2, 3, 4, 7} {8, -1, 7}) -> {8, 3, 7} since 2*3*4*7 = 8*3*7
    template<tensorRank explicitRank, tensorRank implicitRank>
    constexpr Shape<implicitRank> deduceShape(const Shape<explicitRank>& explicitShape, const Shape<implicitRank>& implicitShape);

    namespace Private {
        // Maybe these concepts violate guideline T.20
        // "The intent of concepts is to model semantic categories (Number, Range, RegularFunction)
        // rather than syntactic restrictions (HasPlus, Array)"

        template<template<tensorRank> class Template, tensorRank Rank>
        void derived_from_shape_specialization_impl(const Template<Rank> &);

        /// Concept verifies that 'T' derives from any specialization of Shape
        /// Shape{}, Shape{42}, Shape{1, 2, -1, 4} all satisfy the concept, as does any IndeterminateShape_t
        template <class T>
        concept derived_from_shape = requires(const T& t) {
            derived_from_shape_specialization_impl<Shape>(t);
        };

        /// Specialization of Shape. Used as a flag to denote a shape of rank 'N', but unknown dimensions
        /// Similar to std::dynamic_extent for spans
        template<tensorRank N>
        struct IndeterminateShape_t : public Shape<N> {};
        // Since it extends Shape<N>, it's the size of N tensorDimensions, but the space is wasted.

        /// Concept verifies that 'T' derives from any specialization of Shape
        /// IndeterminateShape_t<N>{}, for any N >= 0 satisfies the concept
        template <class T>
        concept derived_from_indeterminate_shape = requires(const T& t) {
            derived_from_shape_specialization_impl<IndeterminateShape_t>(t);
        };

        /// Concept verifies that 'T' derives from any specialization of Shape
        /// Shape{}, Shape{42}, Shape{1, 2, -1, 4} all satisfy the concept, as does any IndeterminateShape_t
        template <class T>
        concept derived_from_static_shape = derived_from_shape<T> && !derived_from_indeterminate_shape<T>;

        template <tensorRank R>
        requires (R > 0)
        inline constexpr Shape<R-1> ShapeTail(Shape<R> shape){
            return Shape<R - 1>(from_range, shape | std::ranges::views::drop(1));
        }
    }

    /// Create an instance of IndeterminateShape_t<N>
    template <tensorRank N>
    constexpr Private::IndeterminateShape_t<N> IndeterminateShape() { return {}; }
}

#endif //TENSOR_SHAPE_H

#include "TensorII/private/templates/Shape.tpp" // NOLINT(bugprone-suspicious-include)
