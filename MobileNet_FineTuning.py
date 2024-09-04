from keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator를 사용해 데이터를 증강 및 전처리
datagen = ImageDataGenerator(
    rescale=1.0 / 255,         # 이미지를 [0, 1] 범위로 정규화
    rotation_range=20,         # 회전
    width_shift_range=0.2,     # 가로 이동
    height_shift_range=0.2,    # 세로 이동
    shear_range=0.2,           # 전단 변환
    zoom_range=0.2,            # 확대
    horizontal_flip=True,      # 좌우 반전
    fill_mode='nearest'        # 빈 공간을 채우는 방식
)

train_generator = datagen.flow_from_directory(
    'train_data_directory',   # 학습 데이터 경로
    target_size=(224, 224),   # 이미지 크기
    batch_size=32,
    class_mode='categorical'  # 다중 클래스 분류
)