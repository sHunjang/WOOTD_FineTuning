import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms
from keras.models import load_model
import numpy as np

# YOLOv8 모델 로드
model = YOLO('TOP&BOTTOM_Detection.pt')

# 파인튜닝된 MobileNetV2 모델 로드
fine_tuned_model = load_model('best_model.h5')

# MPS 장치 설정 (사용 가능하면 MPS, 그렇지 않으면 CPU)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 이미지 전처리 파이프라인 설정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 탐지할 이미지 경로 리스트 설정
image_paths = [
    os.path.join(os.getcwd(), 'Clothings_Combination/test_1.png'),
    os.path.join(os.getcwd(), 'Wannabe_Fashion/Image_3.png')
]

# YOLOv8 모델로 탐지 수행
results = model(image_paths)

# 탐지된 객체를 저장할 기본 폴더 설정
output_dir = 'detected_objects'
os.makedirs(output_dir, exist_ok=True)

# 각 이미지에 대한 특징 벡터 저장
top_features = []
bottom_features = []

# 각 이미지에 대한 탐지 결과 처리
for idx, result in enumerate(results):
    # 원본 이미지 로드
    img = Image.open(image_paths[idx])
    draw = ImageDraw.Draw(img)  # 이미지에 그리기 위한 객체 생성

    # 탐지된 객체의 정보를 추출
    boxes = result.boxes  # Boxes 객체

    # 탐지된 객체의 바운딩 박스 정보를 사용해 이미지를 크롭 및 저장
    for i, box in enumerate(boxes):
        # MPS 텐서를 CPU로 이동 후 numpy 변환
        xyxy = box.xyxy.cpu().numpy()[0]  # 객체의 바운딩 박스 좌표

        # 바운딩 박스 그리기
        draw.rectangle([xyxy[0], xyxy[1], xyxy[2], xyxy[3]], outline="red", width=3)

        cropped_img = img.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        
        # 이미지가 'RGBA' 모드라면 'RGB'로 변환
        if cropped_img.mode == 'RGBA':
            cropped_img = cropped_img.convert('RGB')
        
        # 크롭된 이미지를 MobileNetV2 모델에 맞게 리사이즈
        resized_img = cropped_img.resize((224, 224))
        
        # 전처리 및 모델에 적용하여 특징 벡터 추출
        input_tensor = preprocess(resized_img).unsqueeze(0).numpy()
        
        # 입력 텐서의 축 순서 변경 (N, C, H, W) -> (N, H, W, C)
        input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))
        
        # 예측 수행
        feature_vector = fine_tuned_model.predict(input_tensor)
        
        # Keras 모델 출력(Numpy 배열)을 PyTorch 텐서로 변환
        feature_vector = torch.tensor(feature_vector).squeeze()  # (1, N) -> (N,)

        # 객체 클래스에 따라 특징 벡터를 저장
        class_id = int(box.cls.cpu().numpy()[0])
        if class_id == 0:
            top_features.append(feature_vector)
        elif class_id == 1:
            bottom_features.append(feature_vector)

    # 탐지된 객체가 그려진 이미지를 저장
    img.save(os.path.join(output_dir, f'detected_{idx}.png'))

# 유사도 측정 함수
def cosine_similarity(feature_list):
    similarities = []
    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            sim = F.cosine_similarity(
                feature_list[i], feature_list[j], dim=0
            )
            similarities.append((i, j, sim.item()))
    return similarities

# 상의끼리 유사도 측정
top_similarities = cosine_similarity(top_features)
print("상의끼리 유사도:")
for i, j, sim in top_similarities:
    print(f"Top_{i}와 Top_{j} 유사도: {sim}")

# 하의끼리 유사도 측정
bottom_similarities = cosine_similarity(bottom_features)
print("하의끼리 유사도:")
for i, j, sim in bottom_similarities:
    print(f"Bottom_{i}와 Bottom_{j} 유사도: {sim}")