from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import time

# 크롬 드라이버 설정
driver = webdriver.Chrome()

# 폴더 생성
folder_name = "musinsa"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# URL 접근
url = "https://www.musinsa.com/categories/item/001?device=mw&sortCode=new"
driver.get(url)
count = 0
actions = driver.find_element(By.CSS_SELECTOR, 'body')
is_first = False

for i in range(1, 11):
    for j in range(1, 4):
        print(f'{i}, {j}')
        try:
            # 요소가 나타날 때까지 기다림
            image_selector = (f'//*[@id="commonLayoutContents"]/div[3]/div/div/div/div[{i}]/div/div[{j}]/div[1]/div/a/div')
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, image_selector)))
            image_box = driver.find_element(By.XPATH, image_selector)
            saveimage_selector = (f'//*[@id="commonLayoutContents"]/div[3]/div/div/div/div[{i}]/div/div[{j}]/div[1]/div/a/div/img')
            saveimage = driver.find_element(By.XPATH, saveimage_selector)
            src = saveimage.get_attribute('src')
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

            # 요소 클릭
            image_box.click()
            driver.implicitly_wait(1)

            try:
                text_selector = ('#root > div.sc-8beamu-0.hOombb > div.sc-of3kep-0.edZdA-D > div.sc-1sxlp32-0.ijnrws')
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, text_selector)))
                text_box = driver.find_element(By.CSS_SELECTOR, text_selector)
                text = text_box.text

                # 추가 작업 수행
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1000);")
                text2_sp = driver.find_element(By.CSS_SELECTOR, '#qnaSection > div.sc-1tp8ac2-0.gtqGwP > div:nth-child(2) > button')
                text2_sp.click()
                text2_list = []
                for k in range(1, 4):
                    text2_selector = (f'#qnaSection > div.sc-1tp8ac2-0.gtqGwP > div.sc-11inhzu-0.jcJLJx > div > div > dl:nth-child({k})')
                    text2_box = driver.find_element(By.CSS_SELECTOR, text2_selector)
                    text2 = text2_box.text
                    text2_list.append(text2)

                all_text = f"{text}\n\n" + "\n".join(text2_list)
                text_file_path = os.path.join(folder_name, f'product_info{count}.txt')

                with open(text_file_path, 'w+', encoding='utf-8') as file:
                    file.write(all_text)
                    print(f'제품 정보가 {text_file_path}에 저장되었습니다.')

                driver.find_element(By.XPATH, '//*[@id="commonLayoutHeader"]/div/button').click()

                if not is_first:
                    actions.send_keys(Keys.ARROW_DOWN)
                    time.sleep(1)
                    actions.send_keys(Keys.ARROW_DOWN)
                    time.sleep(1)
                    actions.send_keys(Keys.ARROW_DOWN)
                    time.sleep(1)
                    is_first = True

            except Exception as e:
                print(f'에러!: {e}')

        except Exception as e:
            print(f'이미지 클릭 에러: {e}')
            actions.send_keys(Keys.ARROW_DOWN)
            time.sleep(1)
            actions.send_keys(Keys.ARROW_DOWN)
            time.sleep(1)
            actions.send_keys(Keys.ARROW_DOWN)
            time.sleep(1)

driver.quit()