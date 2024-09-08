# 필요한 라이브러리 import
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. YOLOv8 모델 로드
model = YOLO('TOP&BOTTOM_Detection.pt')

# 2. 이미지 불러오기
# 두 개의 이미지 경로
image_paths = ['Clothings_Combination/test_1.png', 'Clothings_Combination/test_2.png']
images = [cv2.imread(img) for img in image_paths]

# 3. 탐지 수행
results = [model(image) for image in images]

# 4. 결과 출력 및 시각화
for i, result in enumerate(results):
    # 탐지된 결과를 이미지에 그리기
    result_image = result[0].plot()  # 첫 번째 결과 시각화
    
    # Matplotlib을 사용해 이미지를 화면에 표시
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.title(f"Image {i+1} Detection Results")
    plt.axis('off')
    plt.show()
