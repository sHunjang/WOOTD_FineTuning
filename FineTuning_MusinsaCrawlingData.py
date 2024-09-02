import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV3Large, MobileNetV3Small, MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers.legacy import Adam
from sklearn.metrics.pairwise import cosine_similarity

# MPS 장치 설정 (가능한 경우 GPU, 그렇지 않으면 CPU)
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'

# 하이퍼파라미터 설정
batch_size = 16
epochs_initial = 40
learning_rate_initial = 1e-4
learning_rate_finetune = 1e-5

# 데이터 경로 설정
train_dir = 'Dataset/train'
val_dir = 'Dataset/val'
test_dir = 'Dataset/test'

# 색상 히스토그램 계산 함수
def calculate_color_histogram(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist = np.concatenate([hist_b, hist_g, hist_r])
    return hist

# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # 증가된 회전 범위
    width_shift_range=0.4,  # 증가된 수평 이동 범위
    height_shift_range=0.4,  # 증가된 수직 이동 범위
    shear_range=0.4,  # 증가된 전단 강도
    zoom_range=0.4,  # 증가된 줌 범위
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 트레이닝 및 검증 데이터 생성
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 모델 구성
with tf.device(device):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 색상 히스토그램을 추가적인 입력으로 사용하기 위해 새로운 레이어 생성
    hist_input = tf.keras.Input(shape=(768,))  # 256*3 (RGB 각 채널별 히스토그램 크기)
    combined = Concatenate()([x, hist_input])
    
    # 분류 레이어
    predictions = Dense(train_generator.num_classes, activation='softmax')(combined)
    model = Model(inputs=[base_model.input, hist_input], outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate_initial), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
callbacks = [checkpoint, reduce_lr]

# 사용자 정의 데이터 제너레이터
def custom_data_generator(generator):
    while True:
        x_batch, y_batch = next(generator)
        histograms = np.array([calculate_color_histogram(Image.fromarray((x * 255).astype(np.uint8))) for x in x_batch])
        yield [x_batch, histograms], y_batch

# 모델 학습
with tf.device(device):
    model.fit(
        custom_data_generator(train_generator),
        epochs=epochs_initial,
        validation_data=custom_data_generator(validation_generator),
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks
    )

# 미세 조정
for layer in base_model.layers:
    layer.trainable = True

# 미세 조정 후 재컴파일
with tf.device(device):
    model.compile(optimizer=Adam(learning_rate=learning_rate_finetune), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        custom_data_generator(train_generator),
        epochs=epochs_initial,  # 미세 조정용 추가 에포크 설정 가능
        validation_data=custom_data_generator(validation_generator),
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks
    )

# 모델 저장
model.save('FineTuned_V2_Musinsa_final.keras')

# 테스트 데이터 생성
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # 평가할 때 데이터의 순서가 섞이지 않도록 설정
)

# 모델 평가
with tf.device(device):
    loss, accuracy = model.evaluate(custom_data_generator(test_generator), steps=test_generator.samples // batch_size)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')