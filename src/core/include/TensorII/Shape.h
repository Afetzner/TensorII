//
// Created by Amy Fetzner on 11/24/2023.
//

#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include <type_traits>
#include "TensorII/private/template_util.h"

namespace TensorII::Core {
    using namespace Private;

    using tensorDimension = long long;
    using tensorRank = size_t;
    using tensorSize = size_t;

    template <tensorDimension ... Dimensions>
    struct Shape;

    //region explicit and implicit shape testers
    template<typename>
    struct IsExplicitShape;

    template<>
    struct IsExplicitShape<Shape<>>{
        using value = std::integral_constant<bool, true>;
    };

    template<tensorDimension N>
    struct IsExplicitShape<Shape<N>>{
        using value = std::integral_constant<bool, (N > 0)>;
    };

    template<tensorDimension ... Ns>
    struct IsExplicitShape<Shape<Ns...>>{
        using value = typename Util::all_positive<tensorDimension, Ns...>::value;
    };

    template<typename Shape>
    inline constexpr bool IsExplicitShape_v = IsExplicitShape<Shape>::value::value;

    template <typename Shape>
    concept ExplicitShape = IsExplicitShape_v<Shape>;

    template<typename>
    struct IsImplicitShape;

    template<>
    struct IsImplicitShape<Shape<>>{
        using value = std::integral_constant<bool, false>;
    };

    template<tensorDimension N>
    struct IsImplicitShape<Shape<N>>{
        using value = std::integral_constant<bool, (N == -1)>;
    };

    template<tensorDimension ... Ns>
    struct IsImplicitShape<Shape<Ns...>>{
        using value = std::integral_constant<bool,
                ((Util::count_if_v<Util::is_positive<tensorDimension>, tensorDimension, Ns...> + 1) == sizeof...(Ns))
                && ((Util::count_if_v<Util::is_equal_to<tensorDimension, -1>, tensorDimension, Ns...> == 1))
                >;
    };

    template<typename Shape>
    inline constexpr bool IsImplicitShape_v = IsImplicitShape<Shape>::value::value;

    template <typename Shape>
    concept ImplicitShape = IsImplicitShape_v<Shape>;

    template<typename Shape>
    struct IsValidShape {
        using value = std::integral_constant<bool,
                IsExplicitShape_v<Shape> || IsImplicitShape_v<Shape>
                >;
    };

    template<typename Shape>
    inline constexpr bool IsValidShape_v = IsValidShape<Shape>::value::value;

    template <typename Shape>
    concept ValidShape = IsValidShape_v<Shape>;
    //endregion explicit and implicit shape testers

    //region Shape
    //region explicit shape
    // 0 dimensions
    template<>
    struct Shape<> {
        static constexpr tensorRank rank = 0;
        static constexpr tensorSize size = 1;
    };

    // 1 dimension
    template <tensorDimension first>
        requires (ExplicitShape<Shape<first>>)
    struct Shape<first> {
        static constexpr tensorRank rank = 1;
        static constexpr tensorSize size = first;
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<>;
    };

    // >1 dimensions
    template <tensorDimension first, tensorDimension ... rest>
        requires (ExplicitShape<Shape<first, rest...>>)
    struct Shape<first, rest...> {
        static constexpr tensorRank rank = sizeof...(rest) + 1;
        static constexpr tensorSize size = (first * ... * rest);
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<rest...>;
    };
    //endregion explicit

    //region implicit shape
    // 1 dimension
    template <tensorDimension first>
        requires (ImplicitShape<Shape<first>>)
    struct Shape<first> {
        static constexpr tensorRank rank = 1;
        static constexpr tensorSize knownSize = 1;
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<>;
    };

    // >1 dimensions
    template <tensorDimension first, tensorDimension ... rest>
        requires (ImplicitShape<Shape<first, rest...>>)
    struct Shape<first, rest...> {
        static constexpr tensorRank rank = sizeof...(rest) + 1;
        static constexpr tensorSize knownSize = -(first * ... * rest); // since te only unknown dim is -1, just negate
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<rest...>;
    };
    //endregion implicit shape
    //endregion Shape

    template<ExplicitShape explicitShape, ImplicitShape implicitShape>
    struct DeduceShape {
    private:
        static constexpr tensorDimension deducedDim = explicitShape::size / implicitShape::knownSize;
        static constexpr tensorDimension remainder  = explicitShape::size % implicitShape::knownSize;
    public:
        static constexpr bool deducible = remainder == 0;
        using Shape = typename Util::replace<tensorDimension, -1, deducedDim, implicitShape>::Type;
    };

}

#endif //TENSOR_SHAPE_H
