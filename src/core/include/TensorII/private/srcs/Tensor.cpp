//
// Created by Amy Fetzner on 11/19/2023.
//

#include "TensorII/private/headers/Tensor_private.h"

namespace TensorII::Core {

    template<Scalar DType, auto shape, typename Allocator>
    void Tensor<DType, shape, Allocator>::ArrayDeleter::operator()(std::array<DType, size()> *a) {
        Allocator allocator;
        allocator.deallocate((DType *) a, size());
    }

    template<Scalar DType, auto shape_, typename Allocator>
    Tensor<DType, shape_, Allocator>::Tensor(Private::TensorInitializer<DType, shape_> &&initializer)
            : Tensor<DType, shape_>::Tensor(initializer.values) {}

    //region constructors
    template<Scalar DType, auto shape_, typename Allocator>
    Tensor<DType, shape_, Allocator>::Tensor(typename Private::TensorInitializer<DType, shape_>::Array &array) {
        static_assert(sizeof(Array) == sizeof(DType) * size(),
                      "Mismatch of sizeof(DType[N]) and std::array<DType, N>, cursed compiler?");
        Allocator allocator;
        Array *array_p = new(allocator.allocate(size())) Array;
        data_ = std::unique_ptr<Array, ArrayDeleter>(array_p);
        memcpy_s(data_.get(), size_in_bytes(), array, size_in_bytes());
    }

    template<Scalar DType, auto shape_, typename Allocator>
    constexpr DType *Tensor<DType, shape_, Allocator>::data() noexcept { return data_->data(); }

    template<Scalar DType, auto shape_, typename Allocator>
    constexpr const DType *Tensor<DType, shape_, Allocator>::data() const noexcept { return data_->data(); }
    //endregion constructors

    //region reshape
    template<auto newShape, auto oldShape, Scalar DType>
    requires (oldShape.size() == newShape.size()
              && oldShape.isValidExplicit()
              && newShape.isValidExplicit())
    Tensor<DType, newShape> &
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }

    template<Shape newShape, Shape oldShape, Scalar DType>
    requires (oldShape.size() == newShape.size()
              && oldShape.isValidExplicit()
              && newShape.isValidImplicit())
    Tensor<DType, deduceShape(oldShape, newShape)>&
    reshape(Tensor <DType, oldShape>& t) {
        return t;
    }
    //endregion reshape

    //region 0D tensor specialization
    //region constructors
    template<Scalar DType, typename Allocator>
    Tensor<DType, Shape<0>{}, Allocator>::Tensor(DType value) : data_(value) {}

    template<Scalar DType, typename Allocator>
    Tensor<DType, Shape<0>{}, Allocator>::Tensor(Private::TensorInitializer<DType, Shape<0>{}, 0> &&initializer)
            : Tensor<DType, Shape<0>{}, Allocator>::Tensor(initializer.value) {}
    //endregion constructors

    template<Scalar DType, typename Allocator>
    constexpr DType *Tensor<DType, Shape<0>{}, Allocator>::data() noexcept { return &data_; }

    template<Scalar DType, typename Allocator>
    constexpr const DType *Tensor<DType, Shape<0>{}, Allocator>::data() const noexcept { return &data_; }
    //endregion 0D specialization
}
