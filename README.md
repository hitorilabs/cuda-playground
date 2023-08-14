# Dealing with CUDA Installation

Installing CUDA Toolkit and Driver (Updated 2023/08/13 for 12.2)

These are all the important sections that I missed because I didn't actually read through the guide properly:

- [Delete everything and restart](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver)
- [Network Installation for Ubuntu 22.04](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)
- [Pre-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
- [Package Manager Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
- [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)

# Working with CMake (and pulling in CUTLASS)

Doesn't help that I'm completely new to C++ in general... 

1. Pull down the [cutlass project](https://github.com/NVIDIA/cutlass)

The example in the `cutlass` docs just provided the code:

```c++
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

int main() {

  cutlass::half_t x = 2.25_hf;

  std::cout << x << std::endl;

  return 0;
}
```

However, you actually need to tell cmake that it needs to be compiled with `nvcc` by using the `.cu` file extension instead of `.cpp`, otherwise you will see this mysterious error:

```
[ 50%] Building CXX object CMakeFiles/cuda-playground.dir/cutlass_print_fp16.cpp.o
In file included from /home/bocchi/cuda-playground/../cutlass/include/cutlass/numeric_types.h:100,
                 from /home/bocchi/cuda-playground/cutlass_print_fp16.cpp:3:
/home/bocchi/cuda-playground/../cutlass/include/cutlass/half.h:62:10: fatal error: cuda_fp16.h: No such file or directory
   62 | #include <cuda_fp16.h>
      |          ^~~~~~~~~~~~~
compilation terminated.
make[2]: *** [CMakeFiles/cuda-playground.dir/build.make:76: CMakeFiles/cuda-playground.dir/cutlass_print_fp16.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/cuda-playground.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```

Then in `CMakeLists.txt` you need to include the cutlass headers 

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

cmake_minimum_required(VERSION 3.10)
project(cuda-playground LANGUAGES CUDA CXX)
include_directories($ENV{HOME}/cutlass/include) # or whatever path you pulled cutlass down into
add_executable(cuda-playground cutlass_print_fp16.cu)
```

The convention seems to be making a `build` directory and then `cmake ..` back to the directory with the `CMakeLists.txt`
```sh
mkdir build && cd build
cmake .. -D CMAKE_CUDA_ARCHITECTURES=86
```

Note: running cmake .. regularly now will give you a warning

```
-- Configuring done
CMake Warning (dev) in CMakeLists.txt:
  Policy CMP0104 is not set: CMAKE_CUDA_ARCHITECTURES now detected for NVCC,
  empty CUDA_ARCHITECTURES not allowed.  Run "cmake --help-policy CMP0104"
  for policy details.  Use the cmake_policy command to set the policy and
  suppress this warning.

  CUDA_ARCHITECTURES is empty for target "cuda-playground".
This warning is for project developers.  Use -Wno-dev to suppress it.

CMake Warning (dev) in CMakeLists.txt:
  Policy CMP0104 is not set: CMAKE_CUDA_ARCHITECTURES now detected for NVCC,
  empty CUDA_ARCHITECTURES not allowed.  Run "cmake --help-policy CMP0104"
  for policy details.  Use the cmake_policy command to set the policy and
  suppress this warning.

  CUDA_ARCHITECTURES is empty for target "cuda-playground".
This warning is for project developers.  Use -Wno-dev to suppress it.
```
