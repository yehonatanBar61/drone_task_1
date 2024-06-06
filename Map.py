import os
import numpy as np
from PIL import Image
from Point import Point
import matplotlib.pyplot as plt

class Map:
    def __init__(self, path):
        self.img: Image = Image.open(path).convert('RGB')
        self.map: np.ndarray = self.render_map_from_image_to_boolean().transpose(1,0,2)
        # self.point_map: np.ndarray = self.create_point_map()

    def render_map_from_image_to_boolean(self):
        h, w = self.img.size
        map_array = np.zeros((h, w, 3), dtype=int)
        for y in range(h):
            for x in range(w):
                coordinate = y, x
                r, g, b = self.img.getpixel(coordinate)
                if r != 0 and g != 0 and b != 0:  # consider black
                    map_array[y, x , :] = 255
        return map_array

    def is_black(self, y, x):
        for i in range(3):
            if self.map[y, x, i] == 255:
                return False
        else:
            return True
        
    def create_point_map(self):
        w, h, _ = self.map.shape
        point_map = np.ndarray((w, h))
        for x in range(w):
            for y in range(h):
                point_map[x, y] = Point(f"{x}, {y}", w, 0, h, 0).set_position(x, y)
        
        return point_map


# Example usage:
if __name__ == "__main__":
    path_to_image = "pictures/maps/p11.png"
    my_map = Map(path_to_image)
    plt.imshow(my_map.map)
    plt.show()
