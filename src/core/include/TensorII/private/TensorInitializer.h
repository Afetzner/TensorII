//
// Created by Amy Fetzner on 11/24/2023.
//
#ifndef TENSOR_TENSORINITIALIZER_H
#define TENSOR_TENSORINITIALIZER_H

#include <concepts>
#include <type_traits>
#include <exception>
#include <optional>
#include "TensorII/private/Tensor_predecl.h"
#include "TensorII/Shape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/Types.h"

namespace TensorII::Core::Private {

    template<Scalar, Shape, tensorRank axis = 0>
    struct TensorInitializer;

    //region TensorInitializer Definitions
    // 0 dimensions
    template<Scalar DType>
    class TensorInitializer<DType, Shape<0>{}, 0> {
        template <Scalar, Shape, tensorRank> friend class TensorInitializer;
        friend class TensorII::Core::Tensor<DType, Shape<0>{}>;

        const DType value;
        using Array = DType;

        class Iterator{
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = DType;
            using difference_type   = std::ptrdiff_t;
            using pointer           = const DType *;
            using reference         = const DType &;

            explicit constexpr Iterator(const DType* ptr = nullptr) : m_ptr(ptr) {}
            reference operator*() const { return *m_ptr; }
            pointer operator->() { return m_ptr; }
            Iterator& operator++() { m_ptr++; return *this; }
            Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; } // NOLINT(cert-dcl21-cpp)
            friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
            friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };
        private:
            pointer m_ptr;
        };

    public:
        constexpr TensorInitializer(DType value) // NOLINT(google-explicit-constructor)
        : value(value) {};

        [[nodiscard]]
        constexpr Iterator begin() const { return Iterator{&value}; }
        [[nodiscard]]
        constexpr Iterator end()   const { return Iterator{(&value) + 1}; }
        constexpr size_t size() { return 1; }
};

    template<Scalar DType, Shape shape, tensorRank axis>
    requires (shape.rank() > 0 && axis == shape.rank())
    struct TensorInitializer<DType, shape, axis> {
    private:
        template <Scalar, Shape, tensorRank> friend class TensorInitializer;

        using Array = DType;
        static constexpr const DType* at(const DType& value, const std::array<size_t, shape.rank()>& indecies){
            if (not std::is_constant_evaluated())
                throw std::logic_error("Don't use this at runtime, fool!");
            return &value;
        }
    };

    // >1 dimension
    template<Scalar DType, Shape shape, tensorRank axis>
    requires (shape.rank() > 0 && axis < shape.rank())
    struct TensorInitializer<DType, shape, axis> {
    private:
        template <Scalar, Shape, tensorRank> friend class TensorInitializer;
        friend class Tensor<DType, shape>;

        using LowerArray = typename TensorInitializer<DType, shape, axis + 1>::Array;
        using Array = LowerArray const [shape[axis]];
        const Array& values;

        static constexpr const DType* at(const typename TensorInitializer<DType, shape, axis>::Array& array,
                                   const std::array<size_t, shape.rank()>& indecies){
            if (not std::is_constant_evaluated())
                throw std::logic_error("Don't use this at runtime, fool!");

            // Index the array in the current axis, by the current axis in the indecies array
            // recurse to index the next axis
            return TensorInitializer<DType, shape, axis + 1>::at(array[indecies[axis]], indecies);
            // It's not technically recursion. Also, there's no runtime cost. ¯\_('_')_/¯
        }

        class Iterator{
        public:
            static_assert(axis == 0);
            using end_t = std::integral_constant<bool, true>;
            static constexpr end_t at_end = end_t {};

            using iterator_category = std::random_access_iterator_tag;
            using value_type        = DType;
            using difference_type   = std::ptrdiff_t;
            using pointer           = const DType *;
            using reference         = const DType &;

            explicit constexpr Iterator(const Array* arr = nullptr)
                : underlying(arr)
                , indecies{}
                , ptr(std::is_constant_evaluated() ? at(*underlying, {}) : reinterpret_cast<const DType *>(arr))
                {}

            explicit constexpr Iterator(const Array* arr, end_t)
                : underlying(arr)
                , indecies{}
                , ptr(std::is_constant_evaluated() ? nullptr : reinterpret_cast<const DType *>(arr) + shape.n_elems())
                {}

            constexpr reference operator*() const { return *ptr; }
            constexpr pointer operator->() { return ptr; }

            constexpr Iterator& operator++() {
                if (std::is_constant_evaluated()){
                    // Need to increment the pointer by 1,
                    // In runtime memory model, this is trivial
                    // In consteval, "memory" is not continuous, so I have to explicit index the next element
                    size_t curr_axis;
                    bool keep_going = true;
                    // Imagine a wonky carry-adder, where the maxes of the columns are the shape
                    for(curr_axis = shape.rank(); keep_going and curr_axis-- != 0; /* nothing */) {
                        indecies[curr_axis]++;
                        if (indecies[curr_axis] == shape[curr_axis]) {
                            indecies[curr_axis] = 0;
                        } else {
                            keep_going = false;
                        }
                        ptr = at(*underlying, indecies);
                    }
                    if (keep_going) {
                        ptr = nullptr; // denotes end
                    }
                } else {
                    ptr++;
                }
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
            const Array* underlying;
            std::array<size_t, shape.rank()> indecies;
            pointer ptr;
        };

    public:
        constexpr TensorInitializer(const Array& values) // NOLINT(google-explicit-constructor)
        : values(values) {};

        [[nodiscard]]
        constexpr Iterator begin() const { return Iterator(&values); };
        [[nodiscard]]
        constexpr Iterator end() const { return Iterator(&values, Iterator::at_end); };
        constexpr size_t size() { return shape.n_elems(); }
    };
    //endregion TensorInitializer Definitions
}


#endif //TENSOR_TENSORINITIALIZER_H
