from ultralytics import YOLO
import cv2 as cv
import csv
import random
from pose_keypoint import pose_keypoint
from game_utilities import utilities
from button import button

# run this to run the game

# game variables

current_scene = 0 # 0 = start, 1 = main game, 2 = win

timer_reset_to = 20
game_timer = 0
round_counter = -1

input_name = 0
input_name_old = 0
pose_num = 10

pose_point = []
player_point = []

output_path = "pose_csv/"

button_output = -1

show_bg = True

pose_bag = []


# YOLO setup ---------------------------------

model = YOLO("yolo26n-pose.pt")

# opencv setup ----------------------------------

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

bg = utilities.read_image_alpha("sprites/border.png")

deco1 = utilities.read_image_alpha("sprites/deco01.png")
deco2 = utilities.read_image_alpha("sprites/deco02.png")
title = utilities.read_image_alpha("sprites/title_box01.png")
title_start = button(0, "sprites/ui_box01_small.png")

timer_x = 200
timer_y = 30

draw_line_width = 5
draw_line_color = (255, 100, 100)

timer_sprite = []
for i in range(6):
    sprite = utilities.read_image_alpha("sprites/timer_bar_large.png")
    sprite_partition = int(sprite.shape[1] / 6)

    sprite = sprite[0: sprite.shape[0], i * sprite_partition: (i + 1) * sprite_partition, :]

    sprite = cv.resize(sprite, (0, 0), fx = 0.6, fy = 0.6)
    timer_sprite.append(sprite)

# functions -------------------------------------

def read_csv_pose(file_name, size_config = 1, xy_mod = (0,0)):
    global pose_point

    pose_point = []
    
    with open(output_path + f'{file_name}.csv', mode='r') as file:
        reader = csv.reader(file)

        for i , row in enumerate(reader):
            px, py = row 
            px = int(px)
            py = int(py)
            
            px *= size_config
            py *= size_config
            px += xy_mod[0]
            py += xy_mod[1]

            coord = (int(px), int(py))

            keypoint = pose_keypoint(coord)
            pose_point.append(keypoint)


def read_player_pose(image):
    global player_point

    image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    results = model(image)
    player_point = []

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()

        for num, person in enumerate(keypoints):
            if num == 0:
                for i, kp in enumerate(person):
                    if i == 0 or i >= 5:
                        int_kp = (int(kp[0]), int(kp[1]))
                        player_point.append(int_kp)
                    else:
                        player_point.append(player_point[0])


def verify_keypoints(player_keypoints):
    for i in pose_point:
        i.verify_pose(player_keypoints)

def draw_player_points(image_input, points, draw_line = False):
    img_copy = image_input.copy()
    for i in points:
        img_copy = cv.circle(img_copy, i, 15, (255,0,0), -1)

    # draw lines
    if draw_line == True and len(points) == 17:
        img_copy = cv.line(img_copy, (points[7][0], points[7][1]), (points[9][0], points[9][1]), draw_line_color, draw_line_width)
        img_copy = cv.line(img_copy, (points[8][0], points[8][1]), (points[10][0], points[10][1]), draw_line_color, draw_line_width)

        img_copy = cv.line(img_copy, (points[7][0], points[7][1]), (points[5][0], points[5][1]), draw_line_color, draw_line_width)
        img_copy = cv.line(img_copy, (points[8][0], points[8][1]), (points[6][0], points[6][1]), draw_line_color, draw_line_width)
        
        img_copy = cv.line(img_copy, (points[5][0], points[5][1]), (points[6][0], points[6][1]), draw_line_color, draw_line_width)

        neck_point = (int((points[5][0] + points[6][0]) / 2), int((points[5][1] + points[6][1]) / 2))
        img_copy = cv.line(img_copy, (neck_point[0], neck_point[1]), (points[0][0], points[0][1]), draw_line_color, draw_line_width)

        img_copy = cv.line(img_copy, (points[5][0], points[5][1]), (points[11][0], points[11][1]), draw_line_color, draw_line_width)
        img_copy = cv.line(img_copy, (points[6][0], points[6][1]), (points[12][0], points[12][1]), draw_line_color, draw_line_width)

        img_copy = cv.line(img_copy, (points[11][0], points[11][1]), (points[12][0], points[12][1]), draw_line_color, draw_line_width)
    
    return img_copy

def draw_keypoints(image_input):
    img_copy = image_input.copy()
    for i in pose_point:
        img_copy = i.draw_point(img_copy)

    return img_copy

def refill_bag():
    global pose_bag

    pose_bag = []

    for i in range(11):
        pose_bag.append(i)

def draw_bag():
    global pose_bag

    if len(pose_bag) == 0:
        refill_bag()

    draw_pose = random.choice(pose_bag)
    pose_bag.remove(draw_pose)

    return draw_pose


# main game ----------------------------

input_name = 99
input_name_old = input_name
read_csv_pose(input_name)

test = 0

while True:
    ret, frame = cap.read()

    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (1920, 1080), interpolation = cv.INTER_LINEAR)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

    # do every scene
    read_player_pose(frame)

    # render in every scene
    if show_bg == True:
        frame = utilities.add_image(frame, bg, 0, 0)

    if current_scene == 0:
        # rendering
        frame = utilities.add_image(frame, deco1, 150, 350)
        frame = utilities.add_image(frame, deco2, 1250, 500)
        frame = utilities.add_image(frame, title, 500, 50)
        frame = title_start.draw_box(frame, 700, 300)

        frame = cv.putText(frame, "Stretch Exercise", (580, 180), cv.FONT_HERSHEY_SIMPLEX, 2.7, (0, 0, 0), 8)
        frame = cv.putText(frame, "Start", (800, 380), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)

        button_output = title_start.detect_cursor(player_point)

        if button_output == 0:
            current_scene = 1
            refill_bag()


    elif current_scene == 1:
        # check if points are in place
        verify_keypoints(player_point)

        frame = cv.putText(frame, "Objective", (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv.putText(frame, "Take a pose following", (50, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv.putText(frame, "the guide and hold the pose", (50, 330), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv.putText(frame, "Press Z to toggle bg", (50, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if round_counter == -1:
            frame = cv.putText(frame, "Tutorial", (1370, 100), cv.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)
        else:
            frame = cv.putText(frame, f"Round: {round_counter + 1}/10", (1370, 100), cv.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)

        if game_timer / timer_reset_to <= 0.16:
            frame = utilities.add_image(frame, timer_sprite[0], timer_x, timer_y)

        elif game_timer / timer_reset_to <= 0.32:
            frame = utilities.add_image(frame, timer_sprite[1], timer_x, timer_y)

        elif game_timer / timer_reset_to <= 0.5:
            frame = utilities.add_image(frame, timer_sprite[2], timer_x, timer_y)

        elif game_timer / timer_reset_to <= 0.66:
            frame = utilities.add_image(frame, timer_sprite[3], timer_x, timer_y)

        elif game_timer / timer_reset_to <= 0.82:
            frame = utilities.add_image(frame, timer_sprite[4], timer_x, timer_y)
        
        elif game_timer / timer_reset_to <= 1:
            frame = utilities.add_image(frame, timer_sprite[5], timer_x, timer_y)

        outline = utilities.read_image_alpha(f"outlines/{input_name}.png")
        frame = utilities.add_image(frame, outline, 0, 0)

        frame = draw_keypoints(frame)

        # check if all are in place
        verification = True

        for i in pose_point:
            if i.point_detected == False:
                verification = False
                break

        if verification == True:
                game_timer += 1

        if game_timer >= timer_reset_to:
            game_timer = 0
            round_counter += 1

            if round_counter == 10:
                current_scene = 2
            else:
                input_name = draw_bag()

            
        if input_name_old != input_name:
            read_csv_pose(input_name)
            input_name_old = input_name

    elif current_scene == 2:  
        frame = cv.putText(frame, "Congratulation!", (600, 180), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
        frame = cv.putText(frame, "You finish the stretch!", (600, 300), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
        
        frame = title_start.draw_box(frame, 700, 400)
        frame = cv.putText(frame, "Retry", (800, 480), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)

        button_output = title_start.detect_cursor(player_point)

        if button_output == 0:
            current_scene = 1
            round_counter = 0
            refill_bag()

            input_name = draw_bag()
            input_name_old = input_name
            read_csv_pose(input_name)

    
    # final touches
    frame = draw_player_points(frame, player_point, True)
    frame = cv.resize(frame, (1280, 720), interpolation = cv.INTER_LINEAR)
    cv.imshow('Pose Game', frame)

    # input keys (work in all scenes) ----------------------------

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('a'):
        if input_name > 0:
            input_name -= 1
    if key == ord('d'):
        if input_name < 10:
            input_name += 1
    if key == ord('z'):
        show_bg = not show_bg
    if key == ord('s'):
        game_timer += 5
    if key == ord('0'):
        current_scene = 0
    if key == ord('1'):
        refill_bag()
        current_scene = 1
    if key == ord('2'):
        current_scene = 2
    if key == ord('9'):
        round_counter = 9


cap.release()
cv.destroyAllWindows()