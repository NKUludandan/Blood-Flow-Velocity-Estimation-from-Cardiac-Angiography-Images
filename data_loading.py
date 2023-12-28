import os
import re
import matplotlib.pyplot as plt
from PIL import Image


def numerical_sort(value):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', value)]


def get_img_list(root):
    img_list = []

    for img in sorted(os.listdir(root), key=numerical_sort):
        img_list.append(root + '\\' + img)
    return img_list


if __name__ == '__main__':
    img_list7_orig = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_7\RegisteredSIFT")
    sample_img = Image.open(img_list7_orig[0])
    plt.imshow(sample_img, cmap='gray')  # 736*736
    plt.show()