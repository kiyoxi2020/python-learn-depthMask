#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> forward_reconstructPrevDepth_cuda(
    at::Tensor depth, 
    at::Tensor mv, 
    at::Tensor reconstructDepthImage,
    at::Tensor fDilateDepthImage,
    at::Tensor fDilatedMotionVectorImage,
    int size_h,
    int size_w
);


std::vector<at::Tensor> forward_reconstructPrevDepth(
    at::Tensor depth, 
    at::Tensor mv, 
    at::Tensor reconstructDepthImage,
    at::Tensor fDilateDepthImage,
    at::Tensor fDilatedMotionVectorImage,
    int size_h,
    int size_w){
    CHECK_INPUT(depth);
    CHECK_INPUT(mv);
    CHECK_INPUT(reconstructDepthImage);
    CHECK_INPUT(fDilateDepthImage);
    CHECK_INPUT(fDilatedMotionVectorImage);
    return forward_reconstructPrevDepth_cuda(depth, mv,
                                            reconstructDepthImage,
                                            fDilateDepthImage,
                                            fDilatedMotionVectorImage,
                                            size_h,
                                            size_w);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_reconstructPrevDepth, "FORWARD RECONSTRUCT_PREV_DEPHT (CUDA)");
}