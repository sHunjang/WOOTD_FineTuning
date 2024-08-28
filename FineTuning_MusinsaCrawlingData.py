import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV3Large
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers.legacy import Adam
import os

# MPS 장치 설정 (가능한 경우 GPU, 그렇지 않으면 CPU)
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'

# 하이퍼파라미터 설정
batch_size = 32
epochs_initial = 30
learning_rate_initial = 1e-4
learning_rate_finetune = 1e-5

# 데이터 경로 설정
train_dir = 'Dataset/train'
val_dir = 'Dataset/val'

# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 데이터의 20%를 검증 세트로 사용
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
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # 클래스 수에 맞게 수정
    model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate_initial), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

callbacks = [checkpoint, reduce_lr, early_stopping]

# 모델 학습
with tf.device(device):
    model.fit(
        train_generator,
        epochs=epochs_initial,
        validation_data=validation_generator,
        callbacks=callbacks
    )

# 미세 조정
for layer in base_model.layers:
    layer.trainable = True

# 미세 조정 후 재컴파일
with tf.device(device):
    model.compile(optimizer=Adam(learning_rate=learning_rate_finetune), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_generator,
        epochs=epochs_initial,  # 미세 조정용 추가 에포크 설정 가능
        validation_data=validation_generator,
        callbacks=callbacks
    )

# 모델 저장
model.save('FineTuned_MobileNetV2_final.h5')