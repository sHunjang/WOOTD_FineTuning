import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers.legacy import Adam  # Legacy Adam Optimizer 사용
import numpy as np
import matplotlib.pyplot as plt

# MPS 장치 설정 (가능한 경우 MPS, 그렇지 않으면 CPU)
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'

# 하이퍼파라미터
batch_size = 64
epochs_initial = 30
epochs_finetune = 30
learning_rate_initial = 1e-4
learning_rate_finetune = 1e-5

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=-1))
x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, axis=-1))
x_train = tf.image.resize(x_train, [224, 224])
x_test = tf.image.resize(x_test, [224, 224])
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 모델 구성
with tf.device(device):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate_initial), loss='categorical_crossentropy', metrics=['accuracy'])

# 학습률 찾기 콜백
# def find_learning_rate(model, data, start_lr=1e-6, end_lr=1e-1, epochs=20):
#     K = tf.keras.backend
    
#     class LRFinder(tf.keras.callbacks.Callback):
#         def __init__(self, start_lr, end_lr, steps):
#             super(LRFinder, self).__init__()
#             self.start_lr = start_lr
#             self.end_lr = end_lr
#             self.steps = steps
#             self.lrates = np.geomspace(start_lr, end_lr, steps)
#             self.losses = []

#         def on_train_begin(self, logs=None):
#             self.weights = self.model.get_weights()
#             self.best_loss = float('inf')

#         def on_batch_end(self, batch, logs=None):
#             loss = logs['loss']
#             if not np.isnan(loss) and loss < self.best_loss * 4:
#                 self.best_loss = loss
#                 self.losses.append(loss)
#                 lr = self.lrates[batch % self.steps]
#                 K.set_value(self.model.optimizer.lr, lr)
#             else:
#                 self.model.stop_training = True

#         def on_train_end(self, logs=None):
#             self.model.set_weights(self.weights)
#             min_len = min(len(self.lrates), len(self.losses))
#             lrates_to_plot = self.lrates[:min_len]
#             losses_to_plot = self.losses[:min_len]
#             plt.plot(lrates_to_plot, losses_to_plot)
#             plt.xscale('log')
#             plt.xlabel('Learning rate')
#             plt.ylabel('Loss')
#             plt.show()

#     steps = (len(data[0]) // batch_size) * epochs
#     callback = LRFinder(start_lr, end_lr, steps)
#     model.compile(optimizer=Adam(), loss='categorical_crossentropy')
#     model.fit(data[0], data[1], batch_size=batch_size, callbacks=[callback], epochs=epochs)


# 콜백 설정
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

# # 학습률 찾기 실행
# with tf.device(device):
#     find_learning_rate(model, (x_train, y_train))

# 모델 학습
with tf.device(device):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_initial, validation_data=(x_test, y_test), callbacks=[checkpoint, reduce_lr])

# 미세 조정
for layer in base_model.layers:
    layer.trainable = True

with tf.device(device):
    model.compile(optimizer=Adam(learning_rate=learning_rate_finetune), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_finetune, validation_data=(x_test, y_test), callbacks=[checkpoint, reduce_lr])

# 모델 저장
model.save('Fashion_MNIST_MobileNetV3Small_final.keras')