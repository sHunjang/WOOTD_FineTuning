from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import time

# 크롬 옵션 설정 (헤드리스 모드)
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 창을 띄우지 않음
chrome_options.add_argument("--disable-gpu")  # GPU 비활성화 (옵션)
chrome_options.add_argument("--window-size=1920x1080")  # 창 크기 설정 (필요 시)

# 크롬 드라이버 설정
driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(5)  # 암묵적 대기 추가

# 폴더 생성
folder_name = "T-Shirts"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# URL 접근
url = "https://www.musinsa.com/category/001001?gf=A"
driver.get(url)
count = 0
actions = driver.find_element(By.CSS_SELECTOR, 'body')
is_first = False

def download_image(src, count):
    try:
        image_data = requests.get(src).content
        image_path = os.path.join(folder_name, f'MTM_{count}.jpg')
        with open(image_path, 'wb') as file:
            file.write(image_data)
        print(f'{image_path} 저장 완료')
    except Exception as e:
        print(f'{src}에서 이미지 다운로드 실패: {e}')

while count < 500:
    for i in range(1, 11):
        for j in range(1, 4):
            if count >= 500:
                break  # 350장의 이미지가 저장되면 크롤링을 종료합니다.
            print(f'{i}, {j}')
            try:
                # 이미지 선택자
                image_selector = f'//*[@id="commonLayoutContents"]/div[3]/div/div/div/div[{i}]/div/div[{j}]/div[1]/div/a/div'
                WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, image_selector)))  # 대기 시간 증가
                image_box = driver.find_element(By.XPATH, image_selector)
                saveimage_selector = f'//*[@id="commonLayoutContents"]/div[3]/div/div/div/div[{i}]/div/div[{j}]/div[1]/div/a/div/img'
                saveimage = driver.find_element(By.XPATH, saveimage_selector)
                src = saveimage.get_attribute('src')
                count += 1

                # 이미지 다운로드 및 저장
                if src:
                    download_image(src, count)
                else:
                    print("이미지 URL을 찾을 수 없습니다.")

            except Exception as e:
                print(f'이미지 클릭 에러: {e}')
                # 스크롤을 더 많이 내리고 잠시 대기
                actions.send_keys(Keys.PAGE_DOWN)
                time.sleep(2)

    # 페이지를 아래로 스크롤하여 더 많은 이미지를 로드
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # 페이지가 로드될 시간을 줍니다.

# 드라이버 종료
driver.quit()

print("크롤링이 완료되었습니다.")