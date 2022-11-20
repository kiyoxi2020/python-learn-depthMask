import numpy as np
from reconstructPrevDepth import load_mv, IsOnScreen, reconstructedDepthBilinearWeightThreshold

fDeviceToViewDepth = [1.19209e-7, -1, 0.1]
fTanHalfFOV = 0.99822
DepthClipBaseScale = 4.0


def ConvertFromDeviceToViewSpace(fDeviceDepth):
    return -fDeviceToViewDepth[2] / (fDeviceDepth * fDeviceToViewDepth[1] - fDeviceToViewDepth[0])


def saturate(x):
    if x < 0: return 0
    if x > 1: return 1
    return x


def lerp(x, y, t):
    return x + t * y


def ComputeSampleDepthClip(fPrevDepth, fPrevDepthBilinearWeight, \
                           fCurrentDepthViewSpace, renderSize):
    fPrevDepthViewSpace = abs(ConvertFromDeviceToViewSpace(fPrevDepth))
    fHalfViewportWidth = renderSize[1] * 0.5
    fDepthThreshold = min(fCurrentDepthViewSpace, fPrevDepthViewSpace)
    Ksep = 1.37e-5
    fRequireDepthSeparation = Ksep * fDepthThreshold * fHalfViewportWidth * fTanHalfFOV
    fDepthDiff = fCurrentDepthViewSpace - fPrevDepthViewSpace
    if fDepthDiff > 0:
        fDepthClipFactor = saturate(fRequireDepthSeparation / fDepthDiff)
    else:
        fDepthClipFactor = 1.
    out = fPrevDepthBilinearWeight * fDepthClipFactor * \
        lerp(1, fDepthClipFactor, saturate(fDepthDiff ** 2))
    return out


def ComputeDepthClip(fUvSample, prevDepth, fCurrentDepthViewSpace, renderSize):
    fPxSample = [fUvSample[0] * renderSize[0] - 0.5, \
                 fUvSample[1] * renderSize[1] - 0.5]
    iPxSample = np.floor(fPxSample)
    fPxFrac = [fPxSample[0] - iPxSample[0], \
               fPxSample[1] - iPxSample[1]]
    bilinearWeights = [
        [
            (1 - fPxFrac[0]) * (1 - fPxFrac[1]),
            fPxFrac[0] * (1 - fPxFrac[1])
        ],
        [
            (1 - fPxFrac[0]) * fPxFrac[1],
            fPxFrac[0] * fPxFrac[1]
        ]
    ]
    fDepth = 0.0
    fWeighSum = 0.0
    for y in range(2):
        for x in range(2):
            iSamplePos = iPxSample + (x, y)
            if IsOnScreen(iSamplePos, renderSize):
                fBilinearWeight = bilinearWeights[y][x]
                if (fBilinearWeight > reconstructedDepthBilinearWeightThreshold):
                    fPrevDepth = prevDepth[int(iSamplePos[0])][int(iSamplePos[1])]
                    fDepth += ComputeSampleDepthClip(fPrevDepth, 
                                                     fBilinearWeight, 
                                                     fCurrentDepthViewSpace, renderSize)
                    fWeighSum += fBilinearWeight
    if fWeighSum > 0:
        return fDepth / fWeighSum
    else:
        return DepthClipBaseScale


def func_compute_ij(prevDepth, depth, mv, pos, renderSize, depthClipImage):
    fDepthUv = [(pos[0] + 0.5) / renderSize[0], \
                (pos[1] + 0.5) / renderSize[1]]
    fMotionVector = load_mv(pos, mv)
    fDilatedUv = [fDepthUv[0] + fMotionVector[0], \
                  fDepthUv[1] + fMotionVector[1]]
    fDilatedDepth = depth[pos[0], pos[1]]
    fCurrentDepthViewSpace = abs(ConvertFromDeviceToViewSpace(fDilatedDepth))
    fDepthClip = ComputeDepthClip(fDilatedUv, prevDepth, fCurrentDepthViewSpace, renderSize)
    depthClipImage[pos[0], pos[1]] = fDepthClip


def func_depthClip(recon_prev_depth, dilated_depth, dilated_mv, renderSize):
    deptClipImage = np.zeros(renderSize, dtype=np.float32)
    for i in range(renderSize[0]):
        for j in range(renderSize[1]):
            print(i, j)
            # if i == 100 and j == 100:
            #     print("")
            func_compute_ij(recon_prev_depth, dilated_depth, dilated_mv, [i, j], \
                 renderSize, deptClipImage)
    return deptClipImage