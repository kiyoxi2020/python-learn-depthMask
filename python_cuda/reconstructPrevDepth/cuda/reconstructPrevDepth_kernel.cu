#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// #if  __CUDA_ARCH__ <= 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ float atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
// #endif
static __inline__ __device__ double  atomicMaxDouble(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

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
    // find nearest depth
    int iSampleCount = 9;
    int iSampleOffsets [9][2] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {0, -1},
        {-1, 0},
        {-1, 1},
        {1, 1},
        {-1, -1},
        {1, -1}
    };
    float temp = i / size_w;
    int pos[2] = {int(floor(temp)), int(i - temp * size_w)};
    float depthSample[9];
    for (int iSampleIndex = 0; iSampleIndex < iSampleCount; ++iSampleIndex){
        int iPos[2] = {pos[0] + iSampleOffsets[iSampleIndex][0], 
                       pos[1] + iSampleOffsets[iSampleIndex][1]};
        if (iPos[0] > 0 && iPos[1] > 0 && iPos[0] < size_h && iPos[1] < size_w){
            depthSample[iSampleIndex] = depth[iPos[0] * size_w + iPos[1]];
        }
        else{
            depthSample[iSampleIndex] = 0;
        }
    }
    int fNearestDepthCoord[2] = {pos[0], pos[1]};
    float fNearestDepth = depthSample[0];
    for (int iSampleIndex = 0; iSampleIndex < iSampleCount; ++iSampleIndex){
        int iPos[2] = {pos[0] + iSampleOffsets[iSampleIndex][0], 
                   pos[1] + iSampleOffsets[iSampleIndex][1]};
        if (iPos[0] > 0 && iPos[1] > 0 && iPos[0] < size_h && iPos[1] < size_w){
            float fNdDepth = depthSample[iSampleIndex];
            if (fNdDepth > fNearestDepth){
                fNearestDepth = fNdDepth;
                fNearestDepthCoord[0] = iPos[0];
                fNearestDepthCoord[1] = iPos[1];
            }
        }
    }
    const int iMotionVectorPos[2] = {fNearestDepthCoord[0],
                                 fNearestDepthCoord[1]};
    int iMotionVectorIndex = iMotionVectorPos[0] * size_w + iMotionVectorPos[1];
    float fDilatedMotionVector[2] = {float(mv[iMotionVectorIndex * 2]),
                                    float(mv[iMotionVectorIndex * 2 + 1])};
    // reconstruct prev depth
    float fDepthUv[2] = { (pos[0] + 0.5) / size_h, 
                          (pos[1] + 0.5) / size_w };
    float fPxPrevPos[2] = { (fDepthUv[0] + fDilatedMotionVector[0]) * size_h - 0.5,
                            (fDepthUv[1] + fDilatedMotionVector[1]) * size_w - 0.5};
    int iPxPrevPos[2] = {floor(fPxPrevPos[0]), floor(fPxPrevPos[1])};
    float fPxFrac[2] = {fPxPrevPos[0] - iPxPrevPos[0],
                        fPxPrevPos[1] - iPxPrevPos[1]};
    float bilinearWeights[2][2] = {
        { (1 - fPxFrac[0]) * (1 - fPxFrac[1]), 
          fPxFrac[0] * (1 - fPxFrac[1])},
        { (1 - fPxFrac[0]) * fPxFrac[1], 
          fPxFrac[0] * fPxFrac[1]}
    };
    float reconstructedDepthBilinearWeightThreshold = 0.05;
    for(int y = 0; y < 2; ++y){
        for(int x = 0; x < 2; ++x){
            int offset[2] = {x, y};
            float w = bilinearWeights[y][x];
            if(w > reconstructedDepthBilinearWeightThreshold){
                int storePos[2] = {iPxPrevPos[0] + offset[0],
                                   iPxPrevPos[1] + offset[1]};
                if (storePos[0] > 0 && storePos[1] > 0 && storePos[0] < size_h && storePos[1] < size_w){
                    int index = storePos[0] * size_w + storePos[1];
                    atomicMaxFloat((float *)(&reconstructDepthImage[index]), fNearestDepth);
                    __syncthreads();
                }
            }
        }
    }
    fDilateDepthImage[i] = fNearestDepth;
    fDilatedMotionVectorImage[i * 2] = fDilatedMotionVector[0];
    fDilatedMotionVectorImage[i * 2 + 1] = fDilatedMotionVector[1];
}


std::vector<at::Tensor> forward_reconstructPrevDepth_cuda(
    at::Tensor depth, 
    at::Tensor mv, 
    at::Tensor reconstructDepthImage,
    at::Tensor fDilateDepthImage,
    at::Tensor fDilatedMotionVectorImage,
    int size_h,
    int size_w){
        const int threads = size_w;
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
    // cluster.sync();
    return {reconstructDepthImage, fDilateDepthImage, fDilatedMotionVectorImage};
}

