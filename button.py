import cv2 as cv
import numpy as np
from game_utilities import utilities

class button:
    output_num = 0

    selection_counter_max = 10
    selection_count = 0

    box_sprite = 0
    box_sprite_red = 0

    box_x = 0
    box_y = 0

    box_height = 0
    box_width = 0


    def __init__(self, output_num, box):
        self.output_num = output_num

        self.box_sprite = utilities.read_image_alpha(box)
        self.box_sprite_red = np.zeros(shape=(self.box_sprite.shape[0], self.box_sprite.shape[1], 3), dtype=np.int16)
        self.box_sprite_red[:] = (0, 0, 255)

        self.box_height = self.box_sprite.shape[0]
        self.box_width = self.box_sprite.shape[1]

    def draw_box(self, bg, x, y):
        self.box_x = x
        self.box_y = y

        utilities.add_image(bg, self.box_sprite, self.box_x, self.box_y)

        if self.selection_count > 0:
            utilities.add_image_alpha(bg, self.box_sprite_red, self.box_x, self.box_y, self.selection_count/self.selection_counter_max)

        return bg

    def detect_cursor(self, cursor_array):
        cursor_in = False
        output = -1

        for i in cursor_array:
            if i[0] > self.box_x and i[0] < (self.box_x + self.box_width):
                if i[1] > self.box_y and i[1] < (self.box_y + self.box_height):
                    cursor_in = True
                    break
                
        if cursor_in:
            if self.selection_count < self.selection_counter_max:
                    self.selection_count += 1
            else:
                    output = self.output_num
                    self.selection_count = 0
        else:
            self.selection_count = 0
        
        return output