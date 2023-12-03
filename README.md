# TensorII

TensorII is an open source project aiming to bring massively parallelized tensor computing to native C++. Using template metaprogramming features available in C++20, I aim to being compile time optimizations to tensor computations with an emphasis on lower dimensional tensors (around 4-6) for use in physics and elecromechanical simulations, multispectral data processing and other applications.
The emphasis will be on allowing massive parralelization from the ground up, with plans to automatically target GPUs and multiple processors. This project is inspired by the likes of FTensor, Eigen, NumPy and TensorFlow, but by starting with the aim of parallelization, I hope to be able to make optimizations not possible in other libraries meant to be more flexible.
