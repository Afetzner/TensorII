//
// Created by Amy Fetzner on 11/29/2023.
//

#ifndef TENSOR_TEMPLATE_UTIL_H
#define TENSOR_TEMPLATE_UTIL_H

#include "type_traits"

namespace TensorII::Util {

    //region count_if
    template<typename Condition, typename T, T ... Args>
    struct count_if;

    template<typename Condition, typename T, T Arg, T ... Args>
    struct count_if<Condition, T, Arg, Args...> {
        using count = std::integral_constant<int,
                (Condition::template eval<Arg>::value ? 1 : 0)
                + count_if<Condition, T, Args...>::count::value>;
    };

    template<typename Condition, typename T>
    struct count_if<Condition, T> {
        using count = std::integral_constant<int, 0>;
    };

    template<typename Condition, typename T, T ... Args>
    inline constexpr int count_if_v = count_if<Condition, T, Args...>::count::value;
    //endregion  count_if

    template <typename T, T t>
    struct is_equal {
        template<T Arg>
        using eval = std::integral_constant<bool, (Arg == t)>;
    };

    template <typename T>
    struct is_positive {
        template<T Arg>
        using eval = std::integral_constant<bool, (Arg > 0)>;
    };

    template <typename T>
    struct is_non_negative {
        template<T Arg>
        using eval = std::integral_constant<bool, (Arg >= 0)>;
    };

    template<typename T, T ... Args>
    struct all_positive {
        using value = std::integral_constant<bool,
                count_if_v<is_positive<T>, T, Args...> == sizeof...(Args)>;
    };

    template<typename T, T ... Args>
    inline constexpr bool all_positive_v = all_positive<T, Args...>::value::value;


    static_assert(count_if<is_positive<int>, int, 1, 2, -1, 3, -2>::count::value == 3);
}
#endif //TENSOR_TEMPLATE_UTIL_H
