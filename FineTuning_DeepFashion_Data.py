import tensorflow as tf
from keras.applications import MobileNetV3Small
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers.legacy import Adam  # Legacy Adam Optimizer 사용
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# MPS 장치 설정 (가능한 경우 MPS, 그렇지 않으면 CPU)
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'

# 하이퍼파라미터
batch_size = 64
epochs_initial = 30
epochs_finetune = 30
learning_rate_initial = 1e-4
learning_rate_finetune = 1e-5
num_classes = 50  # 예시로 50개 클래스를 사용

# 데이터 로드 및 전처리
data_dir = '/path/to/deepfashion_dataset/'  # DeepFashion 데이터셋의 경로
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# ImageDataGenerator를 사용하여 데이터 로드 및 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# 모델 구성
with tf.device(device):
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate_initial), loss='categorical_crossentropy', metrics=['accuracy'])

# 학습률 찾기 콜백
def find_learning_rate(model, train_generator, start_lr=1e-6, end_lr=1e-1, epochs=20):
    K = tf.keras.backend
    
    class LRFinder(tf.keras.callbacks.Callback):
        def __init__(self, start_lr, end_lr, steps):
            super(LRFinder, self).__init__()
            self.start_lr = start_lr
            self.end_lr = end_lr
            self.steps = steps
            self.lrates = np.geomspace(start_lr, end_lr, steps)
            self.losses = []

        def on_train_begin(self, logs=None):
            self.weights = self.model.get_weights()
            self.best_loss = float('inf')

        def on_batch_end(self, batch, logs=None):
            loss = logs['loss']
            if not np.isnan(loss) and loss < self.best_loss * 4:
                self.best_loss = loss
                self.losses.append(loss)
                lr = self.lrates[batch % self.steps]
                K.set_value(self.model.optimizer.lr, lr)
            else:
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            self.model.set_weights(self.weights)
            min_len = min(len(self.lrates), len(self.losses))
            lrates_to_plot = self.lrates[:min_len]
            losses_to_plot = self.losses[:min_len]
            plt.plot(lrates_to_plot, losses_to_plot)
            plt.xscale('log')
            plt.xlabel('Learning rate')
            plt.ylabel('Loss')
            plt.show()

    steps = (train_generator.samples // batch_size) * epochs
    callback = LRFinder(start_lr, end_lr, steps)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    model.fit(train_generator, callbacks=[callback], epochs=epochs)

# 콜백 설정
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

# 학습률 찾기 실행
with tf.device(device):
    find_learning_rate(model, train_generator)

# 모델 학습
with tf.device(device):
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs_initial,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[checkpoint, reduce_lr]
    )

# 미세 조정
for layer in base_model.layers:
    layer.trainable = True

with tf.device(device):
    model.compile(optimizer=Adam(learning_rate=learning_rate_finetune), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs_finetune,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[checkpoint, reduce_lr]
    )

# 모델 저장
model.save('DeepFashion_MobileNetV3Small_final.keras')