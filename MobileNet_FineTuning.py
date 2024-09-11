import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers.legacy import Adam  # 하위 호환 Adam Optimizer 사용
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F

# MPS 장치 설정 (MPS가 가능하면 MPS, 그렇지 않으면 CPU 사용)
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'
elif tf.config.list_physical_devices('MPS'):
    device = '/MPS:0'
else:
    device = '/CPU:0'


# 하이퍼파라미터 설정
# •	배치 크기를 늘리면: 학습이 빠르게 진행되며 경사 계산이 안정적이지만, 과적합 위험이 있으며 메모리 사용량이 많아짐
# •	배치 크기를 줄이면: 메모리 사용량이 적고 더 자주 업데이트가 이루어지지만, 학습 속도가 느릴 수 있고 경사 계산이 불안정할 수 있음
batch_size = 32

initial_epochs = 100
# 역할: 초기 학습 단계에서 사전 학습된 모델의 일부를 고정(freeze)하고, 새로 추가된 레이어 또는 최상단 레이어들만 학습하는 기간을 정의
# •	주요 목적:
# •	새로운 데이터에 맞는 레이어 학습: 기존 사전 학습된 모델의 중간 계층을 고정한 상태에서, 새로운 데이터에 맞춰 마지막 레이어(또는 새로 추가한 레이어들)를 학습
# •	모델의 적응: 새로 추가된 레이어들이 빠르게 학습되면서 모델이 현재 문제에 어느 정도 적응하도록 함
# •	안정성: 초기에는 사전 학습된 모델의 가중치를 유지한 상태에서, 새롭게 추가된 부분만 학습하여 안정적으로 시작

fine_tune_epochs = 100
# 역할: 초기 학습이 끝난 후, 사전 학습된 모델의 고정된 레이어를 풀어(unfreeze) 전체 모델을 학습하는 단계 / 전체 모델이 미세하게 조정되는 파인튜닝 단계에서의 학습 기간을 정의
# •	주요 목적:
# •	전체 모델 미세 조정: 파인튜닝 단계에서는 모델의 사전 학습된 가중치도 함께 조정되면서 전체적으로 더 나은 성능을 발휘할 수 있도록 학습
# •	세밀한 성능 향상: 더 작은 학습률로 모델을 미세하게 조정하여, 성능을 추가적으로 개선합니다. 파인튜닝을 통해 모델의 최종 성능 극대화
# •	과적합 방지: 파인튜닝 과정에서 더 작은 학습률을 사용해 학습이 천천히 이루어지므로, 모델이 더 안정적으로 학습되고 과적합을 방지

learning_rate_initial = 1e-5  # 사전 학습된 모델을 일부 고정(freeze)하고, 새로 추가된 레이어를 학습할 때 사용
learning_rate_fine_tune = 1e-5  # 사전 학습된 모델의 고정된 레이어를 해제하고, 모델 전체를 미세 조정할 때 사용

# 데이터 경로 설정
train_dir = 'Dataset/train'  # 학습 데이터 디렉토리
val_dir = 'Dataset/val'  # 검증 데이터 디렉토리
test_dir = 'Dataset/test'  # 테스트 데이터 디렉토리

# 데이터 증강 및 로드 (학습, 검증, 테스트)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # 테스트셋에서는 셔플을 하지 않음
)

# 사전 학습된 MobileNetV2 모델 로드 (ImageNet weights 사용)
with tf.device(device):  # MPS 장치 또는 GPU, CPU에 할당
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 모델 헤드 구성
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='leaky_relu')(x)
    x = Dropout(0.5)(x)  # 과적합 방지를 위한 Dropout 추가
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    # 새로운 모델 정의
    model = Model(inputs=base_model.input, outputs=predictions)

    # 학습을 위한 상단 레이어만 학습 가능하게 설정 (기본 MobileNetV2 가중치는 고정)
    for layer in base_model.layers:
        layer.trainable = True

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=learning_rate_initial),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 콜백 설정 (모델 체크포인트와 학습률 감소)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    # 초기 학습
    model.fit(
        train_generator,
        epochs=initial_epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, reduce_lr]
    )

    # 미세 조정 단계 (기존 모델의 상단 레이어도 학습 가능하게 설정)
    for layer in base_model.layers:
        layer.trainable = False

    # 모델 재컴파일 (더 작은 학습률 사용)
    model.compile(optimizer=Adam(learning_rate=learning_rate_fine_tune),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 재학습 (미세 조정)
    model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, reduce_lr]
    )

    # 테스트 데이터로 모델 평가
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'\n테스트 정확도: {test_acc:.4f}')

    # 테스트 데이터 예측 결과 확인
    y_pred = model.predict(test_generator)
    y_pred_classes = y_pred.argmax(axis=-1)

    # 정답 라벨
    y_true = test_generator.classes

    # 성능 평가 보고서 출력
    print('\n분류 보고서:\n', classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

    # 혼동 행렬 출력
    print('\n혼동 행렬:\n', confusion_matrix(y_true, y_pred_classes))

    # 최종 모델 저장
    model.save('FineTuned_MobileNetV3Large_final.h5')