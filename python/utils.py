import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import matplotlib.pyplot as plt

def read_exr(path, dim, size):
    print("Load file: {}".format(path))
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    data = data[:size[0], :size[1], dim]
    return data


def write_exr(data, path):
    if data.shape[-1] == 2:
        path0 = path[:-4]
        for i in range(2):
            write_exr(data[:, :, i:i+1], path0+"-%d"%(i)+".exr")
    else:
        cv2.imwrite(path, data)


def show_img(data):
    for i in range(len(data.shape[-1])):
        plt.figure()
        plt.imshow(data[:, :, i])
        plt.show()