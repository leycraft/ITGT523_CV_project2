import cv2 as cv
import math

class pose_keypoint:
    coordinate = (0, 0)
    point_detected = False
    measuring_distance = 75

    def __init__(self, new_coord):
        self.coordinate = new_coord

    def draw_point(self, base_image):
        if self.point_detected:
            base_image = cv.circle(base_image, self.coordinate, self.measuring_distance, (0,255,0), 3)
        else:
            base_image = cv.circle(base_image, self.coordinate, self.measuring_distance, (100,100,255), 3)

        return base_image

    def verify_pose(self, player_point_array):
        for i in player_point_array:
            if self.check_distance(i) <= self.measuring_distance:
                self.point_detected = True
                break
            else:
                self.point_detected = False

    def check_distance(self, point):
        distance = math.sqrt(((self.coordinate[0] - point[0]) ** 2) + ((self.coordinate[1] - point[1]) ** 2))
        return distance