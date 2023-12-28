//
// Created by Amy Fetzner on 12/10/2023.
//

#ifndef TENSOR_ANYSHAPE_H
#define TENSOR_ANYSHAPE_H

#include <type_traits>
#include <optional>
#include <ranges>
#include <array>

#include "TensorII/Shape.h"
#include "TensorII/private/ConceptUtil.h"

namespace TensorII::Core {
    template <tensorRank maxRank_>
    class AnyShape{
        template <tensorRank> friend class AnyShape;

        std::array<tensorDimension, maxRank_> dimensions;
        tensorRank currRank;

        /// A constant, random-access, iterator that provides read-only access to the dimensions of an AnyShape
        class ConstIterator {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type        = tensorDimension;
            using difference_type   = std::ptrdiff_t;
            using pointer           = const tensorDimension *;
            using reference         = const tensorDimension &;

            explicit constexpr ConstIterator(const tensorDimension* ptr = nullptr) : m_ptr(ptr) {}
            reference operator*() const { return *m_ptr; }
            pointer operator->() { return m_ptr; }
            ConstIterator& operator++() { m_ptr++; return *this; }
            ConstIterator operator++(int) { ConstIterator tmp = *this; ++(*this); return tmp; } // NOLINT(cert-dcl21-cpp)
            ConstIterator operator+(difference_type diff) { return ConstIterator{m_ptr + diff}; }
            ConstIterator operator-(difference_type diff) { return ConstIterator{m_ptr - diff}; }
            friend bool operator== (const ConstIterator& a, const ConstIterator& b) { return a.m_ptr == b.m_ptr; };
            friend bool operator!= (const ConstIterator& a, const ConstIterator& b) { return a.m_ptr != b.m_ptr; };
        private:
            pointer m_ptr;
        };

        /// A random-access, iterator to the dimensions of an AnyShape
        class Iterator {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type        = tensorDimension;
            using difference_type   = std::ptrdiff_t;
            using pointer           = tensorDimension *;
            using reference         = tensorDimension &;

            explicit constexpr Iterator(tensorDimension* ptr = nullptr) : m_ptr(ptr) {}
            reference operator*() const { return *m_ptr; }
            pointer operator->() { return m_ptr; }
            Iterator& operator++() { m_ptr++; return *this; }
            Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; } // NOLINT(cert-dcl21-cpp)
            Iterator operator+(difference_type diff) { return Iterator{m_ptr + diff}; }
            Iterator operator-(difference_type diff) { return Iterator{m_ptr - diff}; }
            friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
            friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };
        private:
            pointer m_ptr;
        };

    public:
        constexpr ~AnyShape();

        /// Initialize an AnyShape with rank 0
        constexpr AnyShape();

        /// Initialize an AnyShape with dimensions equal to pack arguments and rank equal to the number of args
        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank_)
        constexpr AnyShape(const Dims ... dims); // NOLINT(google-explicit-constructor)

        /// Initialize an AnyShape with same dimensions and rank as a static shape
        template<tensorRank rank_>
        requires(rank_ <= maxRank_)
        constexpr AnyShape(const Shape<rank_>& shape); // NOLINT(google-explicit-constructor)

        /// Initialize an AnyShape with dimensions from a range and rank equal to its length
        /// Requires a sized range. If the n_elems of the range is greater than maxes, throws std::length_error
        template <Util::SizedContainerCompatibleRange<tensorDimension> Range>
        constexpr explicit AnyShape(from_range_t, Range&& range);
        // This could be made to use an un-sized array, I just didn't have a reason to

        /// Set the dimensions of this AnyShape to the pack arguments and the rank equal to the number of args
        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank_)
        AnyShape& emplace(const Dims ... dims);

        /// Set the dimensions of this AnyShape to the dimensions of a range and rank equal to its length
        /// Requires a sized range. If the n_elems of the range is greater than the max, throws std::length_error
        template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
        AnyShape& emplace(from_range_t, Range&& range);
        // This could be made to use an un-sized array, I just didn't have a reason to

        /// Constructs a static shape from the dimensions of this AnyShape.
        /// If the current rank of this AnyShape does not equal the rank of the shape to construct
        template<tensorRank newRank>
        requires(newRank <= maxRank_)
        constexpr Shape<newRank> shape() const;

        /// Creates a new AnyShape, adding the pack arguments to the end of the this AnyShape's dimensions.
        /// If the n_elems of the pack plus the current rank is greater than the max, throws std::length_error
        template <std::convertible_to<tensorDimension> ... Dims>
        [[nodiscard]]
        constexpr AnyShape<maxRank_> augmented(const Dims ... dims) const;

        /// Creates a new AnyShape, adding the pack arguments to the end of this AnyShape's dimensions.
        /// If the n_elems of the pack plus the current rank is greater than the max, throws std::length_error
        template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
        [[nodiscard]]
        constexpr AnyShape<maxRank_> augmented(from_range_t, Range&& range) const;
        // This could be made to use an un-sized array, I just didn't have a reason to

        /// Adds the pack arguments to the end of the AnyShape.
        /// If the n_elems of the pack plus the current rank is greater than the max, throws std::length_error
        template <std::convertible_to<tensorDimension> ... Dims>
        void augment(const Dims ... dims);

        /// Adds the dimensions of the range to the end of this AnyShape.
        /// Requires a sized range.
        /// If the n_elems of the range plus the current rank is greater than the max, throws std::length_error
        template <Util::SizedContainerCompatibleRange<tensorDimension> Range>
        void augment(from_range_t, Range&& range);

        /// Creates a new AnyShape, removing dimensions from this AnyShape on axis greater than 'newRank'.
        /// If the new rank is greater than the the current rank, throws std::length_error
        [[nodiscard]]
        constexpr AnyShape<maxRank_> demoted(tensorRank newRank) const;

        // There could also be a demoted that returns a static shape
        // Right now, it can be done in two steps: AnyShape<M>{}.demoted(N).shape<N>();

        /// Removes dimensions from this shape so the rank becomes 'newRank'.
        /// If the new rank is greater than the the current rank, throws std::length_error
        void demote(tensorRank newRank);

        /// Set the rank of this AnyShape to 0
        constexpr void reset();

        /// Copies the other shape's dimensions and rank into this AnyShape
        template <tensorRank otherRank>
        requires(otherRank <= maxRank_)
        AnyShape& operator=(const Shape<otherRank>& otherShape);

        /// Copies the other AnyShape's dimensions and rank into this AnyShape
        template <tensorRank otherMaxRank>
        AnyShape& operator=(const AnyShape<otherMaxRank>& shape);

        /// This AnyShape == otherShape iff they have the same rank and dimensions
        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& otherShape) const;

        /// This AnyShape == otherAnyShape iff they have the same rank and dimensions
        template <tensorRank otherMaxRank>
        constexpr bool operator==(const AnyShape<otherMaxRank>& otherAnyShape) const;

        /// Returns an r-value to the dimension of this AnyShape at 'axis' given
        /// If 'axis' given is greater than current rank, throws std::out_of_range
        constexpr tensorDimension operator[](tensorRank axis) const;

        /// Returns the maximum rank containable
        [[nodiscard]]
        constexpr tensorRank maxRank() const;

        /// Returns the current rank - the number of dimensions
        [[nodiscard]]
        constexpr tensorRank rank() const;

        /// Returns the current number of dimensions, to be used with std::ranges interface
        /// Not to be confused with the number of elements in the shape: n_elems
        [[nodiscard]]
        constexpr tensorRank size() const;

        /// Returns the number of elements in an array of the shape. I.e. AnyShape{2, 5, 7} has a 2*5*7=70 elements
        /// Not to be confused with the size or rank.
        [[nodiscard]]
        constexpr tensorRank n_elems() const;

        /// Returns if the shape is a valid explicit shape
        /// A shape is a valid explicit shape iff it has only positive dimensions.
        [[nodiscard]]
        constexpr bool isValidExplicit() const;

        /// Returns if the shape is a valid implicit shape
        /// A shape is a valid implicit shape iff it has only one non-positive dimension and it is -1
        [[nodiscard]]
        constexpr bool isValidImplicit() const;

        /// Returns if the shape is a valid explicit shape or a valid implicit shape
        [[nodiscard]]
        constexpr bool isValid() const;

        /// Returns a constant iterator to the shape's dimensions' beginning
        [[nodiscard]]
        constexpr ConstIterator begin() const;;

        /// Returns an iterator to the shape's dimensions' beginning
        constexpr Iterator begin();;

        /// Returns a constant iterator to the shape's dimensions' end
        [[nodiscard]]
        constexpr ConstIterator end() const;;

        /// Returns a iterator to the shape's dimensions' end
        constexpr Iterator end();
    };

    namespace Private{
        template <class T>
        concept derived_from_any_shape = requires(const T& t) {
            derived_from_shape_specialization_impl<AnyShape>(t);
        };

        template <tensorRank R>
        inline constexpr AnyShape<R> ShapeTail(AnyShape<R> shape){
            return AnyShape<R>(from_range, shape | std::ranges::views::drop(1));
        }
    }
}

#endif //TENSOR_ANYSHAPE_H

#include "TensorII/private/templates/AnyShape.tpp" // NOLINT(bugprone-suspicious-include)
