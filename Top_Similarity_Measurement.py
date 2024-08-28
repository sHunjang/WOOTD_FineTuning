import tensorflow as tf
import numpy as np
from keras.applications import MobileNetV3Large, MobileNetV2, MobileNetV3Small
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from numpy.linalg import norm

# 이미지 불러오기 및 전처리
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

# 이미지 경로
image1_path = 'Clothings_Combination/test_1.png'
image2_path = 'Insta_images/test_1.png'

# MobileNetV2 모델 로드 (특징 추출을 위해 fully connected layer는 제외)
model = MobileNetV3Large(weights='imagenet', include_top=False, pooling='avg')

# 두 이미지에 대해 특징 벡터 추출
img1_data = load_and_preprocess_image(image1_path)
img2_data = load_and_preprocess_image(image2_path)

features1 = model.predict(img1_data)
features2 = model.predict(img2_data)

# 코사인 유사도 계산
cosine_similarity = np.dot(features1, features2.T) / (norm(features1) * norm(features2))

print(f'코사인 유사도: {cosine_similarity[0][0]}')