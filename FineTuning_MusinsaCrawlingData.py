import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers.legacy import Adam

# MPS 장치 설정 (가능한 경우 GPU, 그렇지 않으면 CPU)
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'

# 하이퍼파라미터 설정
batch_size = 32
epochs_initial = 50
learning_rate_initial = 1e-4
learning_rate_finetune = 1e-5

# 데이터 경로 설정
train_dir = 'Dataset/train'
val_dir = 'Dataset/val'
test_dir = 'Dataset/test'

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
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate_initial), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

callbacks = [checkpoint, reduce_lr]

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
    ) #괴도키드

# 모델 저장
model.save('FineTuned_Musinsa_final.h5')

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
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')