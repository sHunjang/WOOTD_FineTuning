import requests
from bs4 import BeautifulSoup
import os
import time

# 폴더 생성
folder_name = "musinsa"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# URL 접근
url = "https://www.musinsa.com/category/001005?gf=A"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

count = 0
i = 1

while count < 350:
    items = soup.select('#searchList li')  # 아이템 목록을 선택합니다.
    
    for item in items:
        try:
            # 이미지 URL 추출
            img_tag = item.select_one('div > div.img-block > a > img')
            if img_tag:
                src = img_tag.get('data-original') or img_tag.get('src')
                count += 1

                if src:
                    try:
                        image_data = requests.get(src).content
                        image_path = os.path.join(folder_name, f'image_{count}.jpg')
                        with open(image_path, 'wb') as file:
                            file.write(image_data)
                        print(f'{image_path} 저장 완료')
                    except Exception as e:
                        print(f'{src}에서 이미지 다운로드 실패: {e}')
                else:
                    print("이미지 URL을 찾을 수 없습니다.")

            # 텍스트 정보 수집
            product_info = item.select_one('div > div.article_info > p.list_info')
            if product_info:
                text = product_info.get_text(strip=True)
                text_file_path = os.path.join(folder_name, f'product_info{count}.txt')

                with open(text_file_path, 'w+', encoding='utf-8') as file:
                    file.write(text)
                    print(f'제품 정보가 {text_file_path}에 저장되었습니다.')

        except Exception as e:
            print(f'에러!: {e}')
            continue

    # 다음 페이지로 이동
    i += 1
    next_page_url = f"https://www.musinsa.com/category/001005?gf=A&page={i}"
    response = requests.get(next_page_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    time.sleep(2)  # 페이지 로딩 대기

print("크롤링이 완료되었습니다.")
