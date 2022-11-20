import reconstructPrevDepth.cuda as reconstructPrevDepth_cuda
import torch
import numpy as np

def func_reconstructPrevDepth(depth, mv, renderSize, device="cuda"):
    depth = np.ascontiguousarray(depth)
    mv = np.ascontiguousarray(mv)
    depth = torch.from_numpy(depth).to(device)
    mv = torch.from_numpy(mv).to(device)
    reconstructDepthImage = torch.zeros(renderSize).to(device)
    fDilateDepthImage = torch.zeros(renderSize).to(device)
    fDilatedMotionVectorImage = torch.zeros(renderSize + [2]).to(device)

    reconstructPrevDepth_cuda.forward(depth, mv, 
                                reconstructDepthImage,
                                fDilateDepthImage,
                                fDilatedMotionVectorImage,
                                renderSize[0],
                                renderSize[1])

    # reconstructDepthImage = np.zeros(renderSize, dtype=np.float32)
    # fDilateDepthImage = np.zeros(renderSize, dtype=np.float32)
    # fDilatedMotionVectorImage = np.zeros(renderSize + [2], dtype=np.float32)
    # i, j = 10, 10
    # for i in range(renderSize[0]):
    #     for j in range(renderSize[1]):
    #         print(i, j)
    #         func_compute_ij(depth, mv, [i, j], renderSize, \
    #             reconstructDepthImage, fDilateDepthImage, fDilatedMotionVectorImage)
    reconstructDepthImage = reconstructDepthImage.detach().cpu().numpy()
    fDilateDepthImage = fDilateDepthImage.detach().cpu().numpy()
    fDilatedMotionVectorImage = fDilatedMotionVectorImage.detach().cpu().numpy()
    return reconstructDepthImage, fDilateDepthImage, fDilatedMotionVectorImage