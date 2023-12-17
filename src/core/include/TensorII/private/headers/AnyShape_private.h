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

    template <tensorRank> class AnyShape;

    class AnyShapeIterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = tensorDimension;
        using difference_type   = std::ptrdiff_t;
        using pointer           = tensorDimension *;
        using reference         = tensorDimension &;

        explicit constexpr AnyShapeIterator(tensorDimension* ptr = nullptr) : m_ptr(ptr) {}
        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }
        AnyShapeIterator& operator++() { m_ptr++; return *this; }
        AnyShapeIterator operator++(int) { AnyShapeIterator tmp = *this; ++(*this); return tmp; } // NOLINT(cert-dcl21-cpp)
        friend bool operator== (const AnyShapeIterator& a, const AnyShapeIterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!= (const AnyShapeIterator& a, const AnyShapeIterator& b) { return a.m_ptr != b.m_ptr; };
    private:
        pointer m_ptr;
    };


    template <tensorRank maxRank>
    class AnyShape{
        std::array<tensorDimension, maxRank> dimensions;
        std::optional<tensorRank> currRank;

    public:
        constexpr ~AnyShape();
        constexpr AnyShape();

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank)
        constexpr explicit AnyShape(const Dims ... dims);

        template<tensorRank rank_>
        requires(rank_ <= maxRank)
        constexpr AnyShape(const Shape<rank_>& shape); // NOLINT(google-explicit-constructor)

        template <Util::ContainerCompatibleRange<tensorDimension> Range>
        constexpr explicit AnyShape(from_range_t, Range&& range);

        template<std::convertible_to<tensorDimension> ... Dims>
        requires(sizeof...(Dims) <= maxRank)
        AnyShape& emplace(const Dims ... dims);

        template<Util::ContainerCompatibleRange<tensorDimension> Range>
        AnyShape& emplace(from_range_t, Range&& range);

        template<tensorRank newRank>
        requires(newRank <= maxRank)
        constexpr Shape<newRank> shape() const;

        template <std::convertible_to<tensorDimension> ... Dims>
        [[nodiscard]]
        constexpr AnyShape<maxRank> augmented(const Dims ... dims) const;

        template<Util::ContainerCompatibleRange<tensorDimension> Range>
        [[nodiscard]]
        constexpr AnyShape<maxRank> augmented(from_range_t, Range&& dims) const;

        [[nodiscard]]
        constexpr AnyShape<maxRank> demoted(tensorRank newRank) const;

        template <std::convertible_to<tensorDimension> ... Dims>
        void augment(const Dims ... dims);

        template <Util::ContainerCompatibleRange<tensorDimension> Range>
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
        constexpr AnyShapeIterator begin() const { return AnyShapeIterator{dimensions.data()}; };

        [[nodiscard]]
        constexpr AnyShapeIterator end() const {
            if (currRank.has_value() && currRank.value()) {
                return AnyShapeIterator(dimensions.data() + currRank.value());
            }
            return AnyShapeIterator {dimensions.data()};
        };
    };
}

#endif //TENSOR_ANYSHAPE_H
