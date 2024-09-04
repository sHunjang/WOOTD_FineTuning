import os
from rembg import remove
from PIL import Image, UnidentifiedImageError
import io

# 배경을 제거할 이미지가 있는 폴더 경로
input_folder = "BackGroundRemove/img_bgr/.."

# 배경이 제거된 이미지를 저장할 폴더 경로
output_folder = "BackGroundRemove/output"

# output 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# input 폴더의 각 폴더에 대해 반복
for root, dirs, files in os.walk(input_folder):
    for foldername in dirs:
        folder_path = os.path.join(root, foldername)
        output_subfolder = os.path.join(output_folder, os.path.relpath(folder_path, input_folder))
        # output 폴더에 하위 폴더 생성
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        # 폴더 안의 각 이미지에 대해 작업
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                input_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_subfolder, filename)
                try:
                    # 이미지 열기
                    with open(input_path, "rb") as f:
                        image_data = f.read()
                    # 배경 제거
                    output_data = remove(image_data)
                    # 제거된 이미지를 파일로 저장
                    with open(output_path, "wb") as f:
                        f.write(output_data)
                except UnidentifiedImageError:
                    print(f"Unidentified image file: {input_path}. Skipping.")
                except Exception as e:
                    print(f"Error processing file {input_path}: {e}")

print("배경 제거가 완료되었습니다.")
