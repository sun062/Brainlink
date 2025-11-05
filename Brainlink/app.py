import streamlit as st
import streamlit.components.v1 as components
import os

# --- 페이지 설정 ---
# 페이지 레이아웃을 'wide'로 설정하여 대시보드가 화면을 넓게 쓰도록 합니다.
st.set_page_config(layout="wide")

# --- HTML 파일 경로 설정 ---
# 이 app.py 파일이 위치한 디렉토리를 기준으로 'htmls/index.html' 경로를 생성합니다.
# os.path.dirname(__file__)는 app.py가 있는 폴더의 절대 경로를 반환합니다.
HTML_FILE_PATH = os.path.join(os.path.dirname(__file__), 'htmls', 'index.html')

# --- HTML 파일 읽기 및 표시 ---
try:
    # 지정된 경로에 파일이 있는지 확인합니다.
    if os.path.exists(HTML_FILE_PATH):
        # UTF-8 인코딩으로 HTML 파일을 엽니다.
        with open(HTML_FILE_PATH, 'r', encoding='utf-8') as f:
            html_data = f.read()

        # Streamlit 컴포넌트를 사용하여 HTML 데이터를 렌더링합니다.
        # height=1000으로 설정하고, 대시보드 내용이 길어지면 스크롤이 가능하도록 scrolling=True로 설정합니다.
        st.components.v1.html(html_data, height=1000, scrolling=True)
    
    else:
        # 파일이 존재하지 않을 경우 에러 메시지를 표시합니다.
        st.error(f"오류: 'htmls/index.html' 파일을 찾을 수 없습니다.")
        st.info("app.py와 동일한 위치에 'htmls' 폴더를 만들고, 그 안에 'index.html' 파일을 넣었는지 확인해주세요.")

except Exception as e:
    # 파일을 읽거나 렌더링하는 중 다른 예외가 발생하면 에러를 표시합니다.
    st.error(f"HTML 파일을 로드하는 중 오류가 발생했습니다: {e}")
