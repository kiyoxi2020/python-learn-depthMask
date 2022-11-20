#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void forward_reconstructPrevDepth_cuda_kernel(
    const scalar_t* depth,
    const scalar_t* mv,
    scalar_t* reconstructDepthImage,
    scalar_t* fDilateDepthImage,
    scalar_t* fDilatedMotionVectorImage,
    int size_h,
    int size_w){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size_h * size_w){
        return;
    }
    reconstructDepthImage[i] = depth[i];
    fDilateDepthImage[i] = depth[i];
    fDilatedMotionVectorImage[i * 2] = mv[i * 2];
    fDilatedMotionVectorImage[i * 2 + 1] = mv[i * 2 + 1];
}


std::vector<at::Tensor> forward_reconstructPrevDepth_cuda(
    at::Tensor depth, 
    at::Tensor mv, 
    at::Tensor reconstructDepthImage,
    at::Tensor fDilateDepthImage,
    at::Tensor fDilatedMotionVectorImage,
    int size_h,
    int size_w){
        const int threads = 512;
        const dim3 blocks ((size_h * size_w - 1) / threads + 1);
        AT_DISPATCH_FLOATING_TYPES(depth.type(), "forward_reconstructPrevDepth_cuda", 
            ([&] {
                forward_reconstructPrevDepth_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    depth.data<scalar_t>(),
                    mv.data<scalar_t>(),
                    reconstructDepthImage.data<scalar_t>(),
                    fDilateDepthImage.data<scalar_t>(),
                    fDilatedMotionVectorImage.data<scalar_t>(),
                    size_h,
                    size_w
                );
            }));
    return {reconstructDepthImage, fDilateDepthImage, fDilatedMotionVectorImage};
}

