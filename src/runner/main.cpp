#include <iostream>
#include <cassert>
#include "Tensor/TensorInitializer.h"
#include "Tensor/Tensor.h"


int main() {

    using namespace TensorII::Core;

    Tensor u0 = Tensor<int, Shape<>> (1);
    Tensor t0 = toTensor<int>(1);
    static_assert(std::is_same_v<decltype(u0), decltype(t0)>);
    assert(*t0.data() == 1);

    Tensor u1 = Tensor<int, Shape<3>> ({1, 2, 3});
    Tensor t1 = toTensor<int>({1, 2, 3});
    static_assert(std::is_same_v<decltype(u1), decltype(t1)>);

    TensorInitializer<int, Shape<2, 3>> ({{1, 2, 3},
                                          {4, 5, 6}});

    Tensor u2 = Tensor<int, Shape<2, 3>> ({{1, 2, 3},
                                           {4, 5, 6}});
    Tensor t2 = toTensor<int>({{1, 2, 3},
                               {4, 5, 6}});

    static_assert(std::is_same_v<decltype(u2), decltype(t2)>);

    Tensor u3 = Tensor<int, Shape<2, 2, 3>> ({
        {{1 , 2 , 3 }, {4 , 5 , 6 }},
        {{7 , 8 , 9 }, {10, 11, 12}}
    });
    Tensor t3 = toTensor<int>({
        {{1 , 2 , 3 }, {4 , 5 , 6 }},
        {{7 , 8 , 9 }, {10, 11, 12}}
    });

    static_assert(std::is_same_v<decltype(u3), decltype(t3)>);


    std::cout << "Hello, World!" << std::endl;
}
