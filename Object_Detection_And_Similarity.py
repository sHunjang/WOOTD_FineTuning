import os
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

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

# YOLOv8 모델 로드
yolo_model = YOLO('/Users/seunghunjang/Desktop/WOOTD_Newmodel/TOP&BOTTOM_Detection.pt')

def calculate_color_histogram(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # MobileNet과 동일한 크기로 리사이즈
    hist = img.histogram()  # 색상 히스토그램 계산
    hist = np.array(hist).reshape(3, 256)  # RGB 각 채널 별로 히스토그램 나누기
    hist = hist / np.sum(hist)  # 정규화
    return hist.flatten()

def extract_features(img):
    # 전처리 및 모델에 적용하여 특징 벡터 추출
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = mobilenet_v3(input_tensor).cpu().numpy()
    return feature_vector

def combined_similarity(feature1, feature2, hist1, hist2, alpha=0.5):
    # 모양 기반 유사도 계산
    shape_similarity = cosine_similarity(feature1, feature2)[0][0]

    # 색상 기반 유사도 계산
    color_similarity = cosine_similarity([hist1], [hist2])[0][0]

    # 결합 유사도 계산 (alpha는 가중치 조절 인자)
    combined_similarity = alpha * shape_similarity + (1 - alpha) * color_similarity
    
    return shape_similarity, color_similarity, combined_similarity

# 이미지 파일 경로 설정
image1_path = 'Clothings_Combination/test_13.png'
image2_path = 'Insta_images/test_6.png'

# 색상 기반 유사도 계산
hist1 = calculate_color_histogram(image1_path)
hist2 = calculate_color_histogram(image2_path)
initial_color_similarity = cosine_similarity([hist1], [hist2])[0][0]

print(f"초기 색상 기반 유사도: {initial_color_similarity}")

# 초기 유사도가 0.7 이상인 경우에만 객체 탐지 수행
if initial_color_similarity >= 0.7:
    print("유사도가 0.7 이상이므로 객체 탐지 및 상세 유사도 분석을 수행합니다.")
    
    # 이미지 경로 리스트 생성
    image_paths = [image1_path, image2_path]

    # YOLOv8 모델로 탐지 수행
    results = yolo_model(image_paths)

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
            
            # 특징 벡터 추출
            feature_vector = extract_features(resized_img)

            # 객체 클래스에 따라 특징 벡터를 저장
            if int(box.cls.cpu().numpy()[0]) == 0:
                top_features.append(feature_vector)
            elif int(box.cls.cpu().numpy()[0]) == 1:
                bottom_features.append(feature_vector)

    # 유사도 측정 (모양 + 색상 결합)
    if len(top_features) >= 2:  # 상의 객체가 2개 이상일 때만 유사도 계산
        shape_sim, color_sim, final_sim = combined_similarity(top_features[0], top_features[1], hist1, hist2, alpha=0.5)
        print(f"상의 모양 기반 유사도: {shape_sim}")
        print(f"상의 색상 기반 유사도: {color_sim}")
        print(f"상의 최종 결합 유사도: {final_sim}")

    if len(bottom_features) >= 2:  # 하의 객체가 2개 이상일 때만 유사도 계산
        shape_sim, color_sim, final_sim = combined_similarity(bottom_features[0], bottom_features[1], hist1, hist2, alpha=0.5)
        print(f"하의 모양 기반 유사도: {shape_sim}")
        print(f"하의 색상 기반 유사도: {color_sim}")
        print(f"하의 최종 결합 유사도: {final_sim}")
else:
    print("유사도가 0.7 미만이므로 객체 탐지를 수행하지 않습니다.")