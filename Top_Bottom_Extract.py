from ultralytics import YOLO
import cv2
import os

# Model load
model = YOLO('TOP&BOTTOM_Detection.pt')

# Object Detection
results = model('Data_1.png')

# 결과에서 박스, 클래스, 점수 추출
boxes = results[0].boxes.xyxy.numpy()  # 바운딩 박스 좌표
classes = results[0].boxes.cls.numpy()  # Class ID
scores = results[0].boxes.conf.numpy()  # 신뢰도 점수


# 원본 이미지
image = cv2.imread('Data_1.png')

# 클래스별 저장할 폴더 경로 설정
output_dir = {'Top': 'detected_objects/Top', 'Bottom': 'detected_objects/Bottom'}

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    cropped_img = image[y1:y2, x1:x2]  # Crop the image within the bounding box

    # Assign class name based on the detected class
    class_name = 'Top' if classes[i] == 0 else 'Bottom'
    output_path = os.path.join(output_dir[class_name], f"{class_name}_{i}.jpg")

    # Save the cropped image
    cv2.imwrite(output_path, cropped_img)

print("Detection and saving complete.")