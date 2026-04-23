import cv2 as cv
import math

class pose_keypoint:
    coordinate = (0, 0)
    point_detected = -1
    measuring_distance = 75
    orange_range = 25
    in_orange = False

    def __init__(self, new_coord):
        self.coordinate = new_coord

    def draw_point(self, base_image):
        if self.point_detected == 1:
            base_image = cv.circle(base_image, self.coordinate, self.measuring_distance, (0,255,0), 3)
        elif self.point_detected == 0:
            base_image = cv.circle(base_image, self.coordinate, self.measuring_distance, (0, 224, 255), 3)
        elif self.point_detected == -1:
            base_image = cv.circle(base_image, self.coordinate, self.measuring_distance, (100,100,255), 3)

        return base_image
    
    def draw_point_small(self, base_image, coord):
        if self.point_detected == 1:
            base_image = cv.circle(base_image, coord, 10, (0,255,0), 3)
        elif self.point_detected == 0:
            base_image = cv.circle(base_image, coord, 10, (0, 224, 255), 3)
        elif self.point_detected == -1:
            base_image = cv.circle(base_image, coord, 10, (100,100,255), 3)

        return base_image

    def verify_pose(self, player_point_array):
        self.in_orange = False

        for i in player_point_array:
            if self.check_distance(i) <= self.measuring_distance:
                self.point_detected = 1
                break
            elif self.check_distance(i) <= self.measuring_distance + self.orange_range:
                self.in_orange = True
            else:
                self.point_detected = -1

        if self.in_orange and self.point_detected == -1:
            self.point_detected = 0

    def check_distance(self, point):
        distance = math.sqrt(((self.coordinate[0] - point[0]) ** 2) + ((self.coordinate[1] - point[1]) ** 2))
        return distance