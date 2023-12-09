//
// Created by Amy Fetzner on 11/19/2023.
//
#ifndef TENSOR_TENSOR_1_H
#define TENSOR_TENSOR_1_H

#include <concepts>
#include <memory>
#include <array>
#include "TensorII/StaticShape.h"
#include "TensorII/TensorDType.h"
#include "TensorII/TensorInitializer.h"
#include "TensorII/TensorIndex.h"

namespace TensorII::Core {

    template <Scalar DType>
    using TensorDefaultAllocator = std::allocator<DType>;

    template <Scalar DType, typename Shape_, typename Allocator = TensorDefaultAllocator<DType>>
    class Tensor{
    public:
        using Shape = Shape_;

        static constexpr tensorSize size() noexcept { return Shape::size; };
        static constexpr tensorSize size_in_bytes() noexcept { return size() * sizeof(DType); };

        explicit Tensor(typename TensorInitializer<DType, Shape>::Array&);
        explicit Tensor(TensorInitializer<DType, Shape>&&);

        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

        template<tensorRank rank>
        const DType* operator[](const tensorIndex(&)[rank]) const;
        template<tensorRank rank>
        DType* operator[](const tensorIndex(&)[rank]);

        template<tensorRank rank>
        const DType* operator[](const TensorIndexer<rank>&) const;
        template<tensorRank rank>
        DType* operator[](const TensorIndexer<rank>&);

    private:
        using Array = std::array<DType, size()>;
        struct ArrayDeleter {
            void operator()(std::array<DType, size()>* a);
        };
        std::unique_ptr<Array, ArrayDeleter> data_;
    };

    template <Scalar DType, ExplicitShape OldShape, ExplicitShape NewShape>
            requires (OldShape::size == NewShape::size)
    Tensor<DType, NewShape>&
    reshape(Tensor<DType, OldShape>& t);

    template <Scalar DType, ExplicitShape OldShape, ImplicitShape NewShape>
            requires(DeduceShape<OldShape, NewShape>::deducible)
    Tensor<DType, typename DeduceShape<OldShape, NewShape>::Shape>&
    reshape(Tensor<DType, OldShape>& t);


    // 0D tensor
    template <Scalar DType, typename Allocator>
    class Tensor<DType, Shape<>, Allocator> {
    public:
        using Shape = Shape<>;

        Tensor(DType value); // NOLINT(google-explicit-constructor)
        explicit Tensor(TensorInitializer<DType, Shape>&& initializer);
        Tensor(const Tensor&) = delete;
        Tensor(Tensor&&) = delete;

        static constexpr tensorSize size() noexcept { return Shape::size; };
        static constexpr tensorSize size_in_bytes() noexcept { return size() * sizeof(DType); };
        constexpr DType* data() noexcept;
        constexpr const DType* data() const noexcept;

    private:
        DType data_;
    };

} // TensorII::Core

#endif //TENSOR_TENSOR_1_H
