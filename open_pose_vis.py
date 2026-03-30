import json
import cv2
import os
import numpy as np

img_dir = r"C:\Users\Yifeng Pan\Documents\compsci760\FIFA-Skeletal-Tracking-Starter-Kit-2026\data\images\ARG_FRA_183303"
json_dir = r"C:\Users\Yifeng Pan\Documents\compsci760\openpose_json\ARG_FRA_183303"

files = sorted(os.listdir(json_dir))

for file in files:
    json_path = os.path.join(json_dir, file)
    img_path = os.path.join(img_dir, file.replace("_keypoints.json", ".jpg"))

    img = cv2.imread(img_path)

    with open(json_path) as f:
        data = json.load(f)

    for person in data["people"]:
        keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)

        for x, y, c in keypoints:
            if c > 0.2:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imshow("OpenPose JSON View", img)

    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()