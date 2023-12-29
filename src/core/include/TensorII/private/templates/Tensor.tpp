//
// Created by Amy Fetzner on 11/19/2023.
//

#ifndef TENSOR_TENSOR_TPP
#define TENSOR_TENSOR_TPP

#include "TensorII/Tensor.h"

namespace TensorII::Core {

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor()
        : data_ {}
    {}

//    template<Scalar DType, auto shape_>
//    requires (derived_from_shape<decltype(shape_)>)
//    constexpr Tensor<DType, shape_>::Tensor(typename TensorInitializer<DType, shape_>::Array&& array)
//        : data_ {}
//    {
//        // assert c-array same size as std::array
//        static_assert(sizeof(Array) == sizeof(decltype(array)));
//        if (!std::is_constant_evaluated()){
//            if constexpr (std::is_bounded_array_v<typename std::remove_cvref_t<decltype(array)>>)
//                memmove_s(data_.data(), size_in_bytes(), array, size_in_bytes());
//            else
//                data_[0] = array; // edge case for 0-tensor, where "array" is a value type
//            return;
//        }
//        // The constant expression version is much slower
//        auto initializer = TensorInitializer<DType, shape_>(array);
//        std::ranges::copy_n(initializer.begin(), size(), data_.begin());
//    }

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor(typename TensorInitializer<DType, shape_>::Array&& array)
            : Tensor<DType, shape_>::Tensor(from_range, TensorInitializer<DType, shape_>(array))
    {}

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor(TensorInitializer<DType, shape_>&& initializer)
        : Tensor<DType, shape_>::Tensor(from_range, initializer)
    {} // Use mem-move maybe? I should investigate if its any faster

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    template<Util::ContainerCompatibleRange<DType> Range>
    constexpr Tensor<DType, shape_>::Tensor(from_range_t, Range && range)
        : data_ {}
    {
        std::ranges::copy_n(range.begin(), size(), data_.begin());
    }

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::Tensor(Tensor && other) noexcept
        : data_{}
    {
        if (!std::is_constant_evaluated()){
            memmove_s(data_.data(), size_in_bytes(), other.data(), other.size_in_bytes());
            return;
        }
        for(size_t i = 0; i < Array::size(); i++){
            data_[i] = other.data_[i];
        }
    }

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>& Tensor<DType, shape_>::operator=(Tensor && other) noexcept {
        if (!std::is_constant_evaluated()){
            memmove_s(data_.data(), size_in_bytes(), other.data(), other.size_in_bytes());
            return *this;
        }
        for(size_t i = 0; i < Array::size(); i++){
            data_[i] = other.data_[i];
        }
        return *this;
    }

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::size_type Tensor<DType, shape_>::size() noexcept {
        return shape_.n_elems();
    }

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::size_type Tensor<DType, shape_>::size_in_bytes() noexcept {
        return size() * sizeof(DType);
    }

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::pointer Tensor<DType, shape_>::data() noexcept {
        return data_.data();
    }

    template<Scalar DType, auto shape_>
    requires (derived_from_shape<decltype(shape_)>)
    constexpr Tensor<DType, shape_>::const_pointer Tensor<DType, shape_>::data() const noexcept {
        return data_.data();
    }

    template<auto newShape, auto oldShape, Scalar DType>
    requires (oldShape.n_elems() == newShape.n_elems()
              && oldShape.isValidExplicit()
              && newShape.isValidExplicit())
    Tensor<DType, newShape> &
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }

    template<Shape newShape, Shape oldShape, Scalar DType>
    requires (oldShape.n_elems() == newShape.n_elems()
              && oldShape.isValidExplicit()
              && newShape.isValidImplicit())
    Tensor<DType, deduceShape(oldShape, newShape)>&
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }
}

#endif //TENSOR_TENSOR_TPP
