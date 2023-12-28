import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from data_loading import get_img_list
from pro_vel import nearest_point, a_star

def find_branch_endpoints(image, n):
    image_array = np.array(image)
    image_array = image_array > 0
    start_point = nearest_point(image_array)
    distances = a_star(image_array, start_point)

    endpoints = {}
    rows, cols = len(distances), len(distances[0])

    for i in range(rows):
        for j in range(cols):
            if distances[i][j] >= 0:
                surrounding = np.copy(distances[max(0, i - n // 2):min(rows, i + n // 2 + 1),max(0, j - n // 2):min(cols, j + n // 2 + 1)])
                if np.max(surrounding) == distances[i][j]:
                    endpoints[(i, j)] = distances[i][j]
    return endpoints

def branch_visualization(image,endpoints):
    color = 255
    point_size = 10
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    for point in endpoints:
        draw.rectangle([point[1] - point_size // 2, point[0] - point_size // 2, point[1] + point_size // 2, point[0] + point_size // 2],fill=color)

    plt.imshow(image_copy, cmap='gray')
    plt.show()

def t_endpoints(img1, img2, n):
    endpoints1 = find_branch_endpoints(img1,n)
    endpoints2 = find_branch_endpoints(img2,n)
    print(endpoints1)
    print(endpoints2)

    branch_visualization(img1,endpoints1)
    branch_visualization(img2,endpoints2)


if __name__ == '__main__':
    img_list7_sift = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_7\RegisteredSIFT")
    t_endpoints(Image.open(img_list7_sift[2]), Image.open(img_list7_sift[3]), 70)