import os
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# 바탕화면 경로 설정 및 폴더 경로 지정 (macOS에서 HOME 사용)
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
folder_name = 'Crawlling_folder'  # 저장할 폴더 이름
folder_path = os.path.join(desktop_path, folder_name)

# 폴더가 존재하지 않으면 생성
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 시작 날짜와 종료 날짜 설정
start_date = datetime(2024, 1, 5)
end_date = datetime(2024, 1, 10)

# 현재 날짜 초기화
current_date = start_date

# Chrome 드라이버 경로 설정 (macOS의 경우 chromedriver 설치 경로 확인 필요)
# brew로 설치한 경우: /opt/homebrew/bin/chromedriver
chrome_driver_path = '/path/to/chromedriver'  # chromedriver 설치 경로로 변경하세요

# Selenium 웹 드라이버 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument('headless')  # 브라우저를 표시하지 않음

# 웹드라이버 실행
driver = webdriver.Chrome(executable_path=chrome_driver_path, options=options)

while current_date <= end_date:
    # 날짜 문자열 생성
    year = current_date.strftime('%Y')
    month = current_date.strftime('%m')
    day = current_date.strftime('%d')

    # URL 생성
    main_page_url = f'https://news.naver.com/breakingnews/section/101/260?date={year}{month}{day}'

    driver.get(main_page_url)
    time.sleep(3)  # 페이지가 로드될 때까지 기다림

    # "더보기" 버튼을 클릭하여 추가 기사 로드
    try:
        while True:
            more_button = driver.find_element_by_class_name('section_more_inner')
            more_button.click()
            time.sleep(0.5)  # 기다려야 할 시간을 조정할 수 있음
            
            # 새로운 기사들이 로드될 때까지 기다림
            new_articles_loaded = WebDriverWait(driver, 1).until(
                EC.staleness_of(more_button)
            )
            if not new_articles_loaded:
                break
                
    except:
        pass  # 더 이상 "더보기" 버튼이 없을 때 예외 발생

    # 페이지 소스 가져오기
    main_page_html = driver.page_source

    soup = BeautifulSoup(main_page_html, 'html.parser')

    article_links = []

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if '/mnews/article/' in href:
            if 'comment/' in href:
                href = href.replace('comment/', '')
            article_links.append(href)

    article_links = list(set(article_links))

    for i, link in enumerate(article_links):
        driver.get(link)
        time.sleep(1)  # 페이지가 로드될 때까지 기다림
        html = driver.page_source

        soup = BeautifulSoup(html, 'html.parser')

        target_element = soup.find(class_='go_trans _article_content')

        if target_element:
            target_text = target_element.get_text()

            # 파일을 날짜와 링크 인덱스 기반으로 저장
            file_path = os.path.join(folder_path, f'Crawlling{year}_{month}_{day}_{i}.txt')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(target_text)
            print(f'{year}-{month}-{day} 데이터가 저장되었습니다.')
        else:
            print(f"해당 클래스 이름을 가진 요소를 찾을 수 없습니다. {link}")

    # 다음 날짜로 이동
    current_date += timedelta(days=1)

# 드라이버 종료
driver.quit()