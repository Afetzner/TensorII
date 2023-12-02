#include "TensorII/Tensor.h"


int main() {
    using namespace TensorII::Core;

    Tensor t0 = Tensor<int, Shape<>> (1); // NOLINT(modernize-use-auto)

    Tensor t1 = Tensor<int, Shape<3>> ({1, 2, 3});

    Tensor t2 = Tensor<int, Shape<2, 3>> ({{1, 2, 3},
                                           {4, 5, 6}});
    Tensor t3 = Tensor<int, Shape<2, 2, 3>> ({
                                                     {{1 , 2 , 3 }, {4 , 5 , 6 }},
                                                     {{7 , 8 , 9 }, {10, 11, 12}}
                                             });

    Tensor t4 = Tensor<int, Shape<2, 3, 2>> ({
                                                     {{1 , 2} , {3 , 4 }, {5 , 6 }},
                                                     {{7 , 8} , {9 , 10}, {11, 12}}
                                             });

    Tensor t5 = Tensor<int, Shape<1, 1, 1, 5>> ({{{{1, 2, 3, 4, 5}}}});
}
