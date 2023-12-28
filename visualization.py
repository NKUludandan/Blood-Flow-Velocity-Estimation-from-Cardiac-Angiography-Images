import numpy as np
from PIL import Image, ImageDraw
from data_loading import get_img_list
from pro_vel import nearest_point, a_star

def distance_normalization(distances):
    max_val = np.max(distances[distances > 0])
    distances = np.where(distances > 0, distances/max_val, distances)
    return distances


def map_value_to_color(value):
    if value == -1:
        return (0, 0, 0)  # black
    else:
        # 0-100 to blue-red
        return (int(255*value), 0, int(255*(1-value)))


def distance_visualiztion(image):
    image_array = np.array(image)
    image_array = image_array > 0
    start_point = nearest_point(image_array)
    distances = a_star(image_array, start_point)
    distances = distance_normalization(distances)

    width, height = len(distances[0]), len(distances)
    img = Image.new('RGB', (width, height))

    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = map_value_to_color(distances[y][x])

    img.show()
    img.save('visualization.png')


if __name__ == '__main__':
    img_list7_sift = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_7\RegisteredSIFT")
    image3 = Image.open(img_list7_sift[4])
    distance_visualiztion(image3)