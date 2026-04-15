from ultralytics import YOLO

# run this to get the pose coordinates for the game

model = YOLO("yolo26n-pose.pt")

output_path = "pose_csv/"
image_num = 12

#--------------------------------------

for i in range(image_num):
    results = model(f"poses/{i}.png")
    
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()

        with open(output_path + f"{i}.csv", "w") as f:
            person = keypoints[0]
            for i, kp in enumerate(person):
                if (i >= 7 and i <= 10):
                    f.write(f"{int(kp[0])}, {int(kp[1])}\n")