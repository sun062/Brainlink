import streamlit as st
import os

# --- Streamlit 앱 설정 ---
# 페이지 레이아웃을 넓게 설정하여 대시보드가 잘 보이도록 합니다.
st.set_page_config(
    page_title="뇌파 분석 대시보드 (Streamlit)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# HTML 파일의 **절대 경로**를 안전하게 설정
# 1. __file__을 사용하여 현재 스크립트(app.py)의 경로를 가져옵니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. htmls 폴더와 index.html 파일명을 지정합니다.
HTML_FOLDER_NAME = "htmls"
HTML_FILE_NAME = "index.html"

# 3. os.path.join()으로 디렉토리 경로, 폴더, 파일 이름을 결합하여 절대 경로를 완성합니다.
#    -> BASE_DIR/htmls/index.html 경로가 생성됩니다.
HTML_FILE_PATH = os.path.join(BASE_DIR, HTML_FOLDER_NAME, HTML_FILE_NAME)


def load_html_file(file_path):
    """지정된 경로에서 HTML 파일을 읽어 그 내용을 문자열로 반환합니다."""
    try:
        # 'utf-8' 인코딩을 사용하여 파일 내용을 읽습니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # 경로 디버깅을 위해 로드 시도 경로를 출력합니다.
        st.error(f"오류: HTML 파일을 찾을 수 없습니다. 경로를 확인해주세요: **'{file_path}'**")
        return None
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

# HTML 파일 내용 로드 (수정된 절대 경로 사용)
html_content = load_html_file(HTML_FILE_PATH)

if html_content:
    # Streamlit에 HTML 내용을 임베드합니다.
    st.html(
        html_content, 
        height=2000
    )
    
    st.markdown("""
        ---
        <div style="text-align: center; color: gray;">
            이 대시보드는 Streamlit의 st.html() 기능을 사용하여 
            외부 HTML/CSS/JS 코드를 임베드한 것입니다.
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="text-align: center; color: #DC2626; padding: 20px; border: 1px solid #DC2626; border-radius: 8px;">
            <h3>파일 로드 실패</h3>
            <p>파일 경로를 확인해주세요.</p>
        </div>
    """, unsafe_allow_html=True)

# Streamlit 실행 명령어: streamlit run app.py
