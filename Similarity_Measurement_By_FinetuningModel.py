import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

# MPS 장치 설정 (가능한 경우 MPS, 아니면 CPU)
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'

# 파인튜닝된 MobileNetV3 Small 모델 로드
finetuned_model = load_model('FineTuned_V2_Musinsa_final.h5', compile=False)

# 이미지 전처리 파이프라인 설정
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # 0-1 사이로 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    return img

# YOLOv8 모델 로드
yolo_model = YOLO('/Users/seunghunjang/Desktop/WOOTD_Newmodel/TOP&BOTTOM_Detection.pt')

def calculate_color_histogram(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    
    # OpenCV에서 사용할 수 있도록 RGB에서 BGR로 변환
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 히스토그램 계산 (각 채널에 대해)
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    
    # 히스토그램 정규화
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    
    # 하나의 히스토그램으로 결합
    hist = np.concatenate([hist_b, hist_g, hist_r])
    return hist

def extract_features(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = preprocess_image(img)
    with tf.device(device):
        feature_vector = finetuned_model.predict(img)
    return feature_vector

def combined_similarity(feature1, feature2, hist1, hist2, alpha=0.3):
    shape_similarity = cosine_similarity(feature1, feature2)[0][0]
    
    # 히스토그램 비교 - OpenCV의 히스토그램 비교 함수 사용
    color_similarity = cv2.compareHist(np.array([hist1], dtype='float32'), 
                                       np.array([hist2], dtype='float32'), 
                                       cv2.HISTCMP_CORREL)
    combined_similarity = alpha * shape_similarity + (1 - alpha) * color_similarity
    return shape_similarity, color_similarity, combined_similarity

# 이미지 파일 경로 설정
image1_path = 'Clothings_Combination/test_1.png'
image2_path = 'Insta_images/test_8.png'

# 초기 색상 및 모양 기반 유사도 계산
hist1 = calculate_color_histogram(image1_path)
hist2 = calculate_color_histogram(image2_path)

feature1 = extract_features(Image.open(image1_path))
feature2 = extract_features(Image.open(image2_path))

shape_similarity, color_similarity, initial_combined_similarity = combined_similarity(feature1, feature2, hist1, hist2, alpha=0.5)

print(f"초기 모양 기반 유사도: {shape_similarity*100:.2f}%")
print(f"초기 색상 기반 유사도: {color_similarity*100:.2f}%")
print(f"초기 최종 결합 유사도: {initial_combined_similarity*100:.2f}%")

if initial_combined_similarity*100.0 >= 70.0:
    print("초기 유사도가 70% 이상이므로 객체 탐지 및 상세 유사도 분석을 수행합니다.")
    
    # 이미지 경로 리스트 생성
    image_paths = [image1_path, image2_path]

    # YOLOv8 모델로 탐지 수행
    results = yolo_model(image_paths)

    output_dir = 'detected_objects'
    os.makedirs(output_dir, exist_ok=True)

    top_features = []
    bottom_features = []

    for idx, result in enumerate(results):
        img = Image.open(image_paths[idx])
        boxes = result.boxes  # Boxes 객체

        for i, box in enumerate(boxes):
            xyxy = box.xyxy.cpu().numpy()[0]
            cropped_img = img.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))

            if cropped_img.mode == 'RGBA':
                cropped_img = cropped_img.convert('RGB')

            resized_img = ImageOps.pad(cropped_img, (224, 224), color=(255, 255, 255))

            feature_vector = extract_features(resized_img)

            if int(box.cls.cpu().numpy()[0]) == 0:
                top_features.append(feature_vector)
            elif int(box.cls.cpu().numpy()[0]) == 1:
                bottom_features.append(feature_vector)

    if len(top_features) >= 2:
        shape_sim, color_sim, final_sim = combined_similarity(top_features[0], top_features[1], hist1, hist2, alpha=0.01)
        print(f"상의 모양 기반 유사도: {shape_sim*100:.2f}%")
        print(f"상의 색상 기반 유사도: {color_sim*100:.2f}%")
        print(f"상의 최종 결합 유사도: {final_sim*100:.2f}%")
        print("--------------------------------------------------------")

    if len(bottom_features) >= 2:
        shape_sim, color_sim, final_sim = combined_similarity(bottom_features[0], bottom_features[1], hist1, hist2, alpha=0.01)
        print(f"하의 모양 기반 유사도: {shape_sim*100:.2f}%")
        print(f"하의 색상 기반 유사도: {color_sim*100:.2f}%")
        print(f"하의 최종 결합 유사도: {final_sim*100:.2f}%")
else:
    print("초기 유사도가 0.7 미만이므로 객체 탐지를 수행하지 않습니다.")