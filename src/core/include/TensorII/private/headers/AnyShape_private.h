//
// Created by Amy Fetzner on 12/10/2023.
//

#ifndef TENSOR_ANYSHAPE_PRIVATE_H
#define TENSOR_ANYSHAPE_PRIVATE_H

#include <type_traits>
#include <optional>
#include <ranges>
#include <array>

#include "TensorII/Shape.h"
#include "TensorII/private/ConceptUtil.h"

namespace TensorII::Core {
    template <tensorRank maxRank>
    class AnyShape{
        std::array<tensorDimension, maxRank> dimensions;
        tensorRank currRank;

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
            friend bool operator== (const ConstIterator& a, const ConstIterator& b) { return a.m_ptr == b.m_ptr; };
            friend bool operator!= (const ConstIterator& a, const ConstIterator& b) { return a.m_ptr != b.m_ptr; };
        private:
            pointer m_ptr;
        };

    public:
        constexpr ~AnyShape();
        constexpr AnyShape();

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank)
        constexpr explicit AnyShape(const Dims ... dims);

        template<tensorRank rank_>
        requires(rank_ <= maxRank)
        constexpr AnyShape(const Shape<rank_>& shape); // NOLINT(google-explicit-constructor)

        template <Util::SizedContainerCompatibleRange<tensorDimension> Range>
        constexpr explicit AnyShape(from_range_t, Range&& range);

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank)
        AnyShape& emplace(const Dims ... dims);

        template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
        AnyShape& emplace(from_range_t, Range&& range);

        template<tensorRank newRank>
        requires(newRank <= maxRank)
        constexpr Shape<newRank> shape() const;

        template <std::convertible_to<tensorDimension> ... Dims>
        [[nodiscard]]
        constexpr AnyShape<maxRank> augmented(const Dims ... dims) const;

        template<Util::SizedContainerCompatibleRange<tensorDimension> Range>
        [[nodiscard]]
        constexpr AnyShape<maxRank> augmented(from_range_t, Range&& dims) const;

        [[nodiscard]]
        constexpr AnyShape<maxRank> demoted(tensorRank newRank) const;

        template <std::convertible_to<tensorDimension> ... Dims>
        void augment(const Dims ... dims);

        template <Util::SizedContainerCompatibleRange<tensorDimension> Range>
        void augment(from_range_t, Range&& range);

        void demote(tensorRank newRank);

        constexpr void reset();

        template <tensorRank otherRank>
        requires(otherRank <= maxRank)
        AnyShape& operator=(const Shape<otherRank>& otherShape);

        template <tensorRank otherMaxRank>
        AnyShape& operator=(const AnyShape<otherMaxRank>& shape);

        template <tensorRank otherRank>
        constexpr bool operator==(const Shape<otherRank>& otherShape) const;

        template <tensorRank otherMaxRank>
        constexpr bool operator==(const AnyShape<otherMaxRank>& otherAnyShape) const;

        constexpr tensorDimension operator[](tensorRank i) const;

        [[nodiscard]]
        constexpr tensorRank rank() const;

        [[nodiscard]]
        constexpr tensorRank size() const;

        [[nodiscard]]
        constexpr bool isValidExplicit() const;

        [[nodiscard]]
        constexpr bool isValidImplicit() const;

        [[nodiscard]]
        constexpr bool isValid() const;

        [[nodiscard]]
        constexpr ConstIterator begin() const {
            return ConstIterator{dimensions.data()};
        };

        [[nodiscard]]
        constexpr ConstIterator end() const {
            if (currRank != 0) {
                return ConstIterator(dimensions.data() + currRank);
            }
            return ConstIterator {dimensions.data()};
        };
    };
}

#endif //TENSOR_ANYSHAPE_H
