#include "TensorII/Tensor.h"
#include "cassert"


int main() {
    using namespace TensorII::Core;
    static constexpr int arr[4][3] = {{00, 01, 02},
                                      {10, 11, 12},
                                      {20, 21, 22},
                                      {30, 31, 32}};
    static constexpr int flat[12] = {00, 01, 02,
                                     10, 11, 12,
                                     20, 21, 22,
                                     30, 31, 32};
    Tensor tensor = Tensor<int, Shape{4, 3}> (arr);
    TensorInitializer<int, Shape{4, 3}> ti (arr);
    assert(ti.size() == 12);
    std::array<int, 12> a {};
    std::ranges::copy(ti.begin(), ti.end(), a.begin());
    assert(std::ranges::equal(flat, ti));
}
