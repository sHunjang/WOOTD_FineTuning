import os
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('/Users/seunghunjang/Desktop/WOOTD_Newmodel/TOP&BOTTOM_Detection.pt')

# MobileNetV3 Small 모델 로드
mobilenet_v3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
mobilenet_v3.classifier = torch.nn.Identity()  # 마지막 분류 레이어를 제거하여 특징 추출기로 사용
mobilenet_v3.eval()

# MPS 장치 설정 (사용 가능하면 MPS, 그렇지 않으면 CPU)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
mobilenet_v3 = mobilenet_v3.to(device)

# 이미지 전처리 파이프라인 설정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 탐지할 이미지 경로 리스트 설정
image_paths = [
    os.path.join(os.getcwd(), 'Clothings_Combination/test_1.png'),
    os.path.join(os.getcwd(), 'Insta_images/test_2.png')
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

    # 탐지된 객체의 정보를 추출
    boxes = result.boxes  # Boxes 객체

    # 탐지된 객체의 바운딩 박스 정보를 사용해 이미지를 크롭 및 저장
    for i, box in enumerate(boxes):
        xyxy = box.xyxy.cpu().numpy()[0]  # 객체의 바운딩 박스 좌표
        cropped_img = img.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        
        # 이미지가 'RGBA' 모드라면 'RGB'로 변환
        if cropped_img.mode == 'RGBA':
            cropped_img = cropped_img.convert('RGB')
        
        # 비율을 유지하며 리사이즈하고 패딩을 추가하여 224x224 크기로 변환
        resized_img = ImageOps.pad(cropped_img, (224, 224), color=(255, 255, 255))
        
        # 전처리 및 모델에 적용하여 특징 벡터 추출
        input_tensor = preprocess(resized_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature_vector = mobilenet_v3(input_tensor).cpu().numpy()

        # 객체 클래스에 따라 특징 벡터를 저장
        if int(box.cls.cpu().numpy()[0]) == 0:
            top_features.append(feature_vector)
        elif int(box.cls.cpu().numpy()[0]) == 1:
            bottom_features.append(feature_vector)

# 특징 벡터를 저장
np.save('top_features.npy', np.array(top_features))
np.save('bottom_features.npy', np.array(bottom_features))