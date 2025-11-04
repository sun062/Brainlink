import streamlit as st
import os

# --- Streamlit 앱 설정 ---
# 페이지 레이아웃을 넓게 설정하여 대시보드가 잘 보이도록 합니다.
st.set_page_config(
    page_title="뇌파 분석 대시보드 (Streamlit)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# HTML 파일의 상대 경로 설정
# **수정된 부분: 'htmls/index.html'로 경로 변경**
HTML_FILE_NAME = "htmls/index.html"


def load_html_file(file_path):
    """지정된 경로에서 HTML 파일을 읽어 그 내용을 문자열로 반환합니다."""
    try:
        # 'utf-8' 인코딩을 사용하여 파일 내용을 읽습니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"오류: HTML 파일을 찾을 수 없습니다. 경로를 확인해주세요: '{file_path}'")
        return None
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

# HTML 파일 내용 로드
html_content = load_html_file(HTML_FILE_NAME)

if html_content:
    # Streamlit에 HTML 내용을 임베드합니다.
    # height를 2000px로 설정하여 스크롤 없이 충분히 많은 내용을 표시하도록 했습니다.
    # HTML 내부의 JavaScript/CSS가 모두 포함된 상태로 실행됩니다.
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
            <p><strong>app.py</strong> 파일과 같은 레벨에 <strong>htmls</strong> 폴더가 있고, 
            그 안에 <strong>index.html</strong> 파일이 있는지 확인해주세요.</p>
        </div>
    """, unsafe_allow_html=True)

# Streamlit 실행 명령어: streamlit run app.py
