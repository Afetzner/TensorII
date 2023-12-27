//
// Created by Amy Fetzner on 12/9/2023.
//

#ifndef TENSOR_TENSORVIEW_H
#define TENSOR_TENSORVIEW_H

#include <ranges>

#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/AnyShape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/private/TensorIndex.h"

namespace TensorII::Core {
    using namespace Private;
    using namespace Indexing;

    //region Unspecialized base class - illegal to instantiate
    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index = 0>
    class TensorView {
        static_assert(!derived_from_shape<decltype(apparentShape)>,
                "Use of unspecialized TensorView is prohibited. "
                "Use TensorView<Tensor<DType, Shape{x, y, z}>> or TensorView<Tensor<DType, AnyShape{}>>");

        using element_type = UnderlyingTensor::value_type;
        element_type* source;  // These are here to make my linter happy
        static constexpr tensorRank rank();
        std::array<TensorIndex, std::max(index, tensorRank(1))> indecies;
    };
    //endregion

    //region Static shape specialization
    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
    requires (derived_from_static_shape<decltype(apparentShape)>)
    class TensorView<UnderlyingTensor, apparentShape, index_> {
    //region iter types
    public:
        using element_type = UnderlyingTensor::value_type;
        using value_type = std::remove_cv_t<typename UnderlyingTensor::value_type>;
        using size_type = tensorSize;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
    //endregion iter types

    private:
        template<derived_from_tensor, auto, tensorRank> friend class TensorView;

        element_type* source;

        static constexpr tensorRank index() { return index_; }
        std::array<TensorIndex, std::max(index_, tensorRank(1))> indecies;

        constexpr explicit TensorView(element_type* source)
            : source(source)
        {}

        // TensorView(TensorView<Shape{...}, Shape{a, b, c, ...}>, IndexTriple{N})
        // -> TensorView<Shape{...}, Shape{b, c, ...}>
        template <auto otherApparentShape>
        requires (ShapeTail(otherApparentShape) == apparentShape)
        constexpr TensorView(TensorView<UnderlyingTensor, otherApparentShape, index_>, Single);

        // TensorView(TensorView<Shape{...}, Shape{a, b, c, ...}>, IndexTriple{})
        // -> TensorView<Shape{...}, Shape{a, b, c, ...}>
        template <auto otherApparentShape>
        requires (otherApparentShape == apparentShape)
        constexpr TensorView(TensorView<UnderlyingTensor, otherApparentShape, index_ - 1>, Empty);

        // TensorView(TensorView<Shape{...}, Shape{a, b, c, ...}>, IndexTriple{x, y, z})
        // -> TensorView<Shape{...}, Shape{(y-z)/z, b, c, ...}>
        template <auto otherApparentShape>
        requires (ShapeTail(otherApparentShape) == apparentShape)
        constexpr TensorView(TensorView<UnderlyingTensor, otherApparentShape, index_ - 1>, Triple);

    public:
        constexpr TensorView(Tensor<element_type, apparentShape>& source)  // NOLINT(google-explicit-constructor)
        requires(apparentShape == source.shape());

        static constexpr tensorRank rank() { return apparentShape.rank(); }

        constexpr TensorView<UnderlyingTensor, ShapeTail(apparentShape), index_ + 1>
        operator[](Single single);

        constexpr TensorView<UnderlyingTensor, apparentShape, index_ + 1>
        operator[](Empty empty);

        constexpr TensorView<UnderlyingTensor, DynamicShape<TensorView<UnderlyingTensor, apparentShape, index_>::rank()>, index_ + 1>
        operator[](Triple index);

        template <Triple triple>
        constexpr auto // TensorView<UnderlyingTensor, apparentShape.replace(index_, triple.count(apparentShape[0])), index_ + 1>
        slice();

        Shape<rank()> shape() { return apparentShape; }
    };

    template<Scalar DType, auto shape>
    TensorView(Tensor<DType, shape>) -> TensorView<Tensor<DType, shape>, shape, 0>;

    //end region Static shape specialization

    namespace Private {
        template<derived_from_tensor UnderlyingTensor, tensorRank maxRank>
        class TensorAnyView {
        //region iter types
        public:
            using element_type = UnderlyingTensor::value_type;
            using value_type = std::remove_cv_t<typename UnderlyingTensor::value_type>;
            using size_type = tensorSize;
            using difference_type = std::ptrdiff_t;
            using reference = value_type&;
            using const_reference = const value_type&;
            using pointer = value_type*;
            using const_pointer = const value_type*;
        //endregion iter types

        private:
            element_type* source;
            AnyShape<maxRank> apparentShape;
            constexpr explicit TensorAnyView(element_type* source);

        public:
            inline constexpr tensorRank rank() { return apparentShape.rank(); }

            template <Shape shape>
            constexpr TensorAnyView(Tensor<element_type, shape>& source)  // NOLINT(google-explicit-constructor)
            requires(shape.rank() < maxRank);

            constexpr auto
            operator[](TensorIndex triple) const;

            AnyShape<maxRank> shape() { return apparentShape; }
        };
    }

    template<derived_from_tensor UnderlyingTensor, auto apparentShape, tensorRank index_>
    requires (derived_from_dynamic_shape<decltype(apparentShape)>)
    class TensorView<UnderlyingTensor, apparentShape, index_>
            : public TensorAnyView<UnderlyingTensor, apparentShape.rank()>
    {};
}

#endif //TENSOR_TENSORVIEW_H

#include "private/templates/TensorView.tpp"