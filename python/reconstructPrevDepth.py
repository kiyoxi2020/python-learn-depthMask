import numpy as np
reconstructedDepthBilinearWeightThreshold = 0.05

def IsOnScreen(pos, renderSize):
    return pos[0] >= 0 and pos[1] >= 0 and pos[0] < renderSize[0] and pos[1] < renderSize[1]


def find_nearest_depth(pos, renderSize, depth):
    iSampleCount = 9
    iSampleOffsets = [
        (0, 0),
        (1, 0),
        (0, 1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (1, 1),
        (-1, -1),
        (1, -1)
    ]
    depthSample = []
    for iSampleIndex in range(0, iSampleCount, 1):
        iPos = [pos[0] + iSampleOffsets[iSampleIndex][0], \
                pos[1] + iSampleOffsets[iSampleIndex][1]]
        if IsOnScreen(iPos, renderSize):
            depthSample.append(depth[iPos[0], iPos[1]])
        else:
            depthSample.append(0)
    fNearestDepthCoord = pos
    fNearestDepth = depthSample[0]
    for iSampleIndex in range(1, iSampleCount, 1):
        iPos = [pos[0] + iSampleOffsets[iSampleIndex][0], \
                pos[1] + iSampleOffsets[iSampleIndex][1]]
        if(IsOnScreen(iPos, renderSize)):
            fNdDepth = depthSample[iSampleIndex]
            if fNdDepth > fNearestDepth:
                fNearestDepth = fNdDepth
                fNearestDepthCoord = iPos
    return fNearestDepth, fNearestDepthCoord


def load_mv(pos, mv):
    return mv[pos[0], pos[1]]

def store_reconstruct_depth(pos, depth, reconstructDepthImage):
    pos = np.int16(pos)
    if depth > reconstructDepthImage[pos[0], pos[1]]:
        reconstructDepthImage[pos[0], pos[1]] = depth

def reconstruct_prev_depth(pos, fDepth, fMotionVector, renderSize, reconstructDepthImage):
    fDepthUv = [(pos[0] + 0.5) / renderSize[0], \
                (pos[1] + 0.5) / renderSize[1]]
    fPxPrevPos = [(fDepthUv[0] + fMotionVector[0]) * renderSize[0] - 0.5, \
                  (fDepthUv[1] + fMotionVector[1]) * renderSize[1] - 0.5]
    iPxPrevPos = np.floor(fPxPrevPos)
    fPxFrac = [fPxPrevPos[0] - iPxPrevPos[0], \
               fPxPrevPos[1] - iPxPrevPos[1]]

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

    for y in range(2):
        for x in range(2):
            offset = (x, y)
            w = bilinearWeights[y][x]
            if w > reconstructedDepthBilinearWeightThreshold:
                storePos = iPxPrevPos + offset
                if IsOnScreen(storePos, renderSize):
                    store_reconstruct_depth(storePos, fDepth, reconstructDepthImage)

def func_compute_ij(depth, mv, pos, renderSize, \
    reconstructDepthImage, fDilateDepthImage, fDilatedMotionVectorImage):
    fDilatedDepth, iNearestDepthCoord = \
        find_nearest_depth(pos, renderSize, depth)
    iMotionVectorPos = iNearestDepthCoord
    fDilatedMotionVector = load_mv(iMotionVectorPos, mv)
    reconstruct_prev_depth(pos, fDilatedDepth, fDilatedMotionVector, renderSize, reconstructDepthImage)
    fDilateDepthImage[pos[0], pos[1]] = fDilatedDepth
    fDilatedMotionVectorImage[pos[0], pos[1]] = fDilatedMotionVector
    return 1


def func_reconstructPrevDepth(depth, mv, renderSize):
    reconstructDepthImage = np.zeros(renderSize, dtype=np.float32)
    fDilateDepthImage = np.zeros(renderSize, dtype=np.float32)
    fDilatedMotionVectorImage = np.zeros(renderSize + [2], dtype=np.float32)
    i, j = 10, 10
    for i in range(renderSize[0]):
        for j in range(renderSize[1]):
            print(i, j)
            func_compute_ij(depth, mv, [i, j], renderSize, \
                reconstructDepthImage, fDilateDepthImage, fDilatedMotionVectorImage)
    return reconstructDepthImage, fDilateDepthImage, fDilatedMotionVectorImage