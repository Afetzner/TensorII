//
// Created by Amy Fetzner on 12/2/2023.
//

#ifndef TENSOR_TENSORINDEXER_H
#define TENSOR_TENSORINDEXER_H

#include "Shape.h"

namespace TensorII::Core {
    using tensorIndex = long long;

    template <tensorRank rank>
    class TensorIndexer {
    public:
        using Indecies = const tensorIndex(&)[rank];
        constexpr TensorIndexer(Indecies indecies) : indecies(indecies) {} // NOLINT(google-explicit-constructor)

    private:
        Indecies indecies;
    };
}

#endif //TENSOR_TENSORINDEXER_H
