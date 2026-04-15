import random
import cv2 as cv

class utilities:
    def read_image_alpha(img_path):
        img = cv.imread(img_path, -1)
        img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

        return img
    
    def add_image(bg, fg, x, y):
        fg_height = fg.shape[0]
        fg_width = fg.shape[1]
        fg_alpha = fg[:,:,3] / 255.0

        for i in range(3):
            bg[y: fg_height + y,x: fg_width + x, i] = fg_alpha * fg[:,:,i] + ((1 - fg_alpha) * bg[y: fg_height + y,x: fg_width + x, i])

        return bg
    
    def add_image_alpha(bg, fg, x, y, a):
        # alpha from 0 to 1.0

        fg_height = fg.shape[0]
        fg_width = fg.shape[1]

        for i in range(3):
            bg[y: fg_height + y,x: fg_width + x, i] = a * fg[:,:,i] + (1 - a) * bg[y: fg_height + y,x: fg_width + x, i]

        return bg