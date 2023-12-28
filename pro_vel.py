import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from data_loading import get_img_list


def nearest_point(arr):
    max_distance = len(arr) + len(arr[0])
    for distance in range(1, max_distance + 1):
        for i in range(distance):
            if arr[i][distance - i] == 1:
                return (i, distance - i)
    return None


def heuristic(point, goal):
    return abs(point[0] - goal[0]) + abs(point[1] - goal[1])


def a_star(map_array, start_point):
    rows, cols = len(map_array), len(map_array[0])
    distances = [[float('inf')] * cols for _ in range(rows)]

    min_heap = [(0, start_point)]
    distances[start_point[0]][start_point[1]] = 0

    while min_heap:
        distance, (row, col) = heapq.heappop(min_heap)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and map_array[new_row][new_col] == 1:
                new_distance = distances[row][col] + 1
                if new_distance < distances[new_row][new_col]:
                    distances[new_row][new_col] = new_distance
                    priority = new_distance + heuristic((new_row, new_col), start_point)
                    heapq.heappush(min_heap, (priority, (new_row, new_col)))

    distances = np.array(distances)
    distances[distances == float('inf')] = -1

    return distances


def predict_velocity(img1, img2):

    image_array1 = np.array(img1)
    image_array2 = np.array(img2)
    image_array1 = image_array1 > 0
    image_array2 = image_array2 > 0

    start_point1 = nearest_point(image_array1)
    distances1 = a_star(image_array1, start_point1)

    start_point2 = nearest_point(image_array2)
    distances2 = a_star(image_array2, start_point2)

    max_distance1 = np.max(distances1)
    max_distance2 = np.max(distances2)

    velocity = abs(max_distance1-max_distance2)
    print("The predicted blood flow velocity is", velocity, 'pixels/frame')

    return velocity

def draw_end_point(image):
    image_array = np.array(image)
    image_array = image_array > 0
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    start_point = nearest_point(image_array)

    distances = a_star(image_array, start_point)
    max_distance = np.max(distances)
    print("Max D:", max_distance)

    max_index = np.argmax(distances)
    max_coord = np.unravel_index(max_index, distances.shape)
    point = (max_coord[1], max_coord[0])

    color = 255
    point_size = 20

    draw.rectangle([point[0] - point_size // 2, point[1] - point_size // 2, point[0] + point_size // 2, point[1] + point_size // 2], fill=color)

    plt.imshow(image_copy, cmap='gray')
    plt.show()


if __name__ == '__main__':
    img_list7_sift = get_img_list(".\AngioData\AngioData\ImsegmentedPt_02 V_7\RegisteredSIFT")
    image1 = Image.open(img_list7_sift[2])
    image2 = Image.open(img_list7_sift[3])
    draw_end_point(image1)
    draw_end_point(image2)
    velocity = predict_velocity(image1, image2)