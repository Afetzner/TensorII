//
// Created by Amy Fetzner on 11/24/2023.
//
#ifndef TENSOR_TENSORINITIALIZER_H
#define TENSOR_TENSORINITIALIZER_H

#include <concepts>
#include <type_traits>
#include <exception>
#include <optional>
#include <format>
#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/Types.h"

namespace TensorII::Core::Private {

    // Returns a DTypes array of that shape. I.e. long and Shape{2, 3, 7} -> long[2][3][7]
    template <typename DType, auto shape>
    requires (derived_from_shape<decltype(shape)>)
    struct ShapeToArray {
        using type = ShapeToArray<DType, ShapeTail(shape)>::type [shape[0]];
    };

    template <typename DType>
    struct ShapeToArray<DType, Shape{}> {
        using type = DType;
    };

    // End of recursion: when the index value is a value type and not an array
    template <typename DType, typename NotArray, size_t rank>
    requires(std::is_same_v<DType, NotArray>)
    constexpr const DType* at(const DType* value,
                              const std::array<size_t, rank>&,
                              tensorRank = 0){
        if (not std::is_constant_evaluated())
            throw std::logic_error("You really, REALLY, shouldn't use this function at runtime");
        return value;
    }

    // Recurse while the indexed value is still an array
    template <typename DType, typename Array, size_t rank>
    constexpr const DType* at(const Array* array,
                              const std::array<size_t, rank>& indecies,
                              tensorRank axis = 0)
    requires(std::is_array_v<Array>)
    {
        if (not std::is_constant_evaluated())
            throw std::logic_error("You really, REALLY, shouldn't use this function at runtime");

        // Index the array in the current axis, by the current axis in the indecies array
        // recurse to index the next axis
        return at<DType, std::remove_extent_t<Array>, rank>(&((*array)[indecies[axis]]), indecies, axis + 1);
        // It's not recursion at runtime at least. ¯\_('_')_/¯
        // Sorry, compilers of the world

        // &((*array)[indecies[axis]]) not array[indecies[axis]]
        // Array is a pointer to an array (not an array pointer), so must be dereferenced before indexing
        // It's a pointer to array to make it more uniform in that you always reference it when you pass to at()
        // Otherwise you'd have to selectively reference if it's a value type, but not if it's an array.
    }


    template<Scalar DType, Shape shape>
    class TensorInitializer{
        friend class Tensor<DType, shape>;

        using SubArray = ShapeToArray<DType, shape>::type;
        // For 0-tensor initializer, just use a DType instead of a DType array
        using Array = std::conditional_t<std::is_array_v<SubArray>, const SubArray&, SubArray>;
        using ArrayPtr = const SubArray*;
        Array values;

        class Iterator {
        public:
            using end_t = std::integral_constant<bool, true>;
            static constexpr end_t at_end = end_t {};

            using iterator_category = std::forward_iterator_tag;
            using value_type        = DType;
            using difference_type   = std::ptrdiff_t;
            using pointer           = const DType *;
            using reference         = const DType &;

            explicit constexpr Iterator(ArrayPtr arr = nullptr)
                    : underlyingArray(arr)
                    , indecies{}
                    , ptr(std::is_constant_evaluated()
                        ? at<DType, SubArray, shape.rank()>(underlyingArray, {})
                        : reinterpret_cast<const DType *>(arr))
            {}

            explicit constexpr Iterator(ArrayPtr arr, end_t)
                    : underlyingArray(arr)
                    , indecies{}
                    , ptr(std::is_constant_evaluated()
                        ? nullptr
                        : reinterpret_cast<const DType *>(arr) + shape.n_elems())
            {}

            constexpr reference operator*() const { return *ptr; }
            constexpr pointer operator->() { return ptr; }

            constexpr Iterator& operator++() {
                if (std::is_constant_evaluated()){
                    // Need to increment the pointer by 1,
                    // In runtime memory model, this is trivial
                    // In consteval, "memory" is not continuous, so I have to explicit index the next element
                    size_t curr_axis;
                    // Imagine a wonky carry-adder, where the maxes of the columns are the shape
                    for(curr_axis = shape.rank(); curr_axis-- != 0; /* nothing */) {
                        ++indecies[curr_axis];
                        if (indecies[curr_axis] == shape[curr_axis]) {
                            indecies[curr_axis] = 0;
                        } else {
                            break;
                        }
                    }
                    if (curr_axis == -1) {
                        ptr = nullptr; // denotes end
                    } else {
                        // Index the array by the indecies in indecies,
                        // i.e. indecies = {2, 3, 7} -> ptr = &underlyingArray[2][3][7]
                        ptr = at<DType, SubArray, shape.rank()>(underlyingArray, indecies);
                    }
                    return *this;
                }
                ++ptr;
                return *this;
            }

            constexpr Iterator operator++(int) {  // NOLINT(cert-dcl21-cpp)
                Iterator tmp = *this;
                ++(*this); return tmp;
            }

            constexpr friend bool operator== (const Iterator& a, const Iterator& b) {
                return a.ptr == b.ptr;
            };

            constexpr friend bool operator!= (const Iterator& a, const Iterator& b) {
                return a.ptr != b.ptr;
            };

        private:
            ArrayPtr underlyingArray;
            std::array<size_t, shape.rank()> indecies;
            pointer ptr;
        };

    public:
        constexpr TensorInitializer(Array values) // NOLINT(google-explicit-constructor)
            : values(values)
        {};

        [[nodiscard]]
        constexpr Iterator begin() const { return Iterator(&values); };
        [[nodiscard]]
        constexpr Iterator end() const { return Iterator(&values, Iterator::at_end); };
        constexpr size_t size() { return shape.n_elems(); }
    };
}


#endif //TENSOR_TENSORINITIALIZER_H
