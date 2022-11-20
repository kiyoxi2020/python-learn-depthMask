import torch
import os
from reconstructPrevDepth import func_reconstructPrevDepth
from depthClip import func_depthClip
from utils import read_exr, write_exr


def main():
    data_path = "../data/dump1/"
    data_out_path = "../data/dump1_out/"
    if not os.path.exists(data_out_path):
        os.mkdir(data_out_path)
    depth_path = os.path.join(data_path, "depth.exr")
    mv_path = os.path.join(data_path, "velocity.exr")
    lr_size = [281, 763]
    depth = read_exr(depth_path, [0], lr_size)
    write_exr(depth, os.path.join(data_out_path, "depth.exr"))
    mv = read_exr(mv_path, [1, 2], lr_size)
    write_exr(mv, os.path.join(data_out_path, "mv.exr"))
    recon_prev_depth, dilated_depth, dilated_mv = \
        func_reconstructPrevDepth(depth, mv, lr_size)
    write_exr(recon_prev_depth, os.path.join(data_out_path, "recon_depth.exr"))
    write_exr(dilated_depth, os.path.join(data_out_path, "dilated_depth.exr"))
    write_exr(dilated_mv, os.path.join(data_out_path, "dilated_mv.exr"))
    depth_clip = \
        func_depthClip(recon_prev_depth, dilated_depth, dilated_mv, lr_size)
    write_exr(depth_clip, os.path.join(data_out_path, "depth_clip.exr"))
    return


if __name__ == "__main__":
    main()