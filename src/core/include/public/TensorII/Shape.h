//
// Created by Amy Fetzner on 11/24/2023.
//

#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include <type_traits>
#include "TensorII/impl/template_util.h"

namespace TensorII::Core {

    using tensorDimension = long long;
    using tensorRank = size_t;
    using tensorSize = size_t;

    template <tensorDimension ... Dimensions>
    struct Shape;

    //region explicit and implicit shape
    template<typename>
    struct ExplicitShape;

    template<>
    struct ExplicitShape<Shape<>>{
        using value = std::integral_constant<bool, true>;
    };

    template<tensorDimension N>
    struct ExplicitShape<Shape<N>>{
        using value = std::integral_constant<bool, (N > 0)>;
    };

    template<tensorDimension ... Ns>
    struct ExplicitShape<Shape<Ns...>>{
        using value = Util::all_positive<tensorDimension, Ns...>::value;
    };

    template<typename Shape>
    inline constexpr bool ExplicitShape_v = ExplicitShape<Shape>::value::value;


    template<typename>
    struct ImplicitShape;

    template<>
    struct ImplicitShape<Shape<>>{
        using value = std::integral_constant<bool, false>;
    };

    template<tensorDimension N>
    struct ImplicitShape<Shape<N>>{
        using value = std::integral_constant<bool, (N == -1)>;
    };

    template<tensorDimension ... Ns>
    struct ImplicitShape<Shape<Ns...>>{
        using value = std::integral_constant<bool,
                ((Util::count_if_v<Util::is_positive<tensorDimension>, tensorDimension, Ns...> + 1) == sizeof...(Ns))
                && ((Util::count_if_v<Util::is_equal<tensorDimension, -1>, tensorDimension, Ns...> == 1))
                >;
    };

    template<typename Shape>
    inline constexpr bool ImplicitShape_v = ImplicitShape<Shape>::value::value;

    template<typename Shape>
    struct ValidShape {
        using value = std::integral_constant<bool,
                ExplicitShape_v<Shape> || ImplicitShape_v<Shape>
                >;
    };

    template<typename Shape>
    inline constexpr bool ValidShape_v = ValidShape<Shape>::value::value;

    //endregion explicit and implicit shape

    // 0 dimensions
    template<>
    struct Shape<> {
        static constexpr tensorRank rank = 0;
        static constexpr tensorSize size = 1;
    };

    // 1 dimension
    template <tensorDimension first>
        requires (ExplicitShape_v<Shape<first>>)
    struct Shape<first> {
        static constexpr tensorRank rank = 1;
        static constexpr tensorSize size = first;
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<>;
    };

    // >1 dimensions
    template <tensorDimension first, tensorDimension ... rest>
        requires (ExplicitShape_v<Shape<first, rest...>>)
    struct Shape<first, rest...> {
        static constexpr tensorRank rank = sizeof...(rest) + 1;
        static constexpr tensorSize size = (first * ... * rest);
        static constexpr tensorDimension first_dim = first;
        using ShapeRemaining = Shape<rest...>;
    };
}

#endif //TENSOR_SHAPE_H
