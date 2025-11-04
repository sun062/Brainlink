import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io

# --- 1. 상수 및 설정 (JS에서 Python으로 포팅) ---

# 페이지 레이아웃을 'wide'로 설정
st.set_page_config(layout="wide", page_title="Brainwave Lite 분석기")

# 뇌파 색상 및 이름 설정
WAVE_COLORS = {
    'Delta': {'name': "델타 (깊은 수면)", 'color': 'rgb(239, 68, 68)', 'groupColor': 'rgb(239, 68, 68)'},
    'Theta': {'name': "세타 (졸음/이완)", 'color': 'rgb(59, 130, 246)', 'groupColor': 'rgb(59, 130, 246)'},
    'LowAlpha': {'name': "저주파 알파 (휴식)", 'color': 'rgb(245, 158, 11)', 'groupColor': 'rgb(16, 185, 129)'},
    'HighAlpha': {'name': "고주파 알파 (평온)", 'color': 'rgb(16, 185, 129)', 'groupColor': 'rgb(16, 185, 129)'},
    'LowBeta': {'name': "저주파 베타 (집중/인지)", 'color': 'rgb(139, 92, 246)', 'groupColor': 'rgb(139, 92, 246)'},
    'HighBeta': {'name': "고주파 베타 (스트레스)", 'color': 'rgb(249, 115, 22)', 'groupColor': 'rgb(139, 92, 246)'},
    'LowGamma': {'name': "저주파 감마 (고차인지)", 'color': 'rgb(107, 114, 128)', 'groupColor': 'rgb(31, 41, 55)'},
    'MidGamma': {'name': "고주파 감마 (통합)", 'color': 'rgb(31, 41, 55)', 'groupColor': 'rgb(31, 41, 55)'},
    'Attention': {'name': "집중도", 'color': 'rgb(22, 163, 74)'},
    'Relaxation': {'name': "안정도", 'color': 'rgb(147, 51, 234)'}
}

CHART_ORDER = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MidGamma']

# 컬럼 매핑 (JS의 로직을 Python으로 구현)
# 비-알파벳/숫자 문자를 제거하여 'attention', 'lowalpha' 등과 같은 순수 키를 생성
COMPRESSED_COLUMN_MAP_KEYS = {
    'attention': 'Attention', '注意力': 'Attention',
    'relaxation': 'Relaxation', '放松度': 'Relaxation',
    'delta': 'Delta', 'δ波': 'Delta',
    'theta': 'Theta', 'θ波': 'Theta',
    'lowalpha': 'LowAlpha', '低α波': 'LowAlpha',
    'highalpha': 'HighAlpha', '高α波': 'HighAlpha',
    'lowbeta': 'LowBeta', '低β波': 'LowBeta',
    'highbeta': 'HighBeta', '高β波': 'HighBeta',
    'lowgamma': 'LowGamma', '低γ波': 'LowGamma',
    'midgamma': 'MidGamma', '高γ波': 'MidGamma',
    'date': 'Date', '日期': 'Date',
    'duration': 'Duration', '时长': 'Duration', '초': 'Duration',
    'tag': 'Tag', '备注': 'Tag',
}

STANDARD_COLUMN_MAP_KEYS = {
    'time': 'Time', '시간': 'Time', 'timestamp': 'Time', '타임스탬프': 'Time',
    'attention': 'Attention', '주의력': 'Attention',
    'relaxation': 'Relaxation', '이완': 'Relaxation',
    'delta': 'Delta', '델타': 'Delta',
    'theta': 'Theta', '세타': 'Theta',
    'lowalpha': 'LowAlpha', '저주파알파': 'LowAlpha',
    'highalpha': 'HighAlpha', '고주파알파': 'HighAlpha',
    'lowbeta': 'LowBeta', '저주파베타': 'LowBeta',
    'highbeta': 'HighBeta', '고주파베타': 'HighBeta',
    'lowgamma': 'LowGamma', '저주파감마': 'LowGamma',
    'midgamma': 'MidGamma', '고주파감마': 'MidGamma',
}

# --- 2. 헬퍼 함수 (파싱 및 계산 로직) ---

def clean_header(header):
    """JS의 cleanHeader와 유사하게, 헤더에서 비-알파벳/숫자 문자를 제거합니다."""
    if not isinstance(header, str):
        header = str(header)
    # 한글, 중국어, 특수 문자 등을 모두 제거하고 영문 소문자/숫자만 남김
    return re.sub(r'[^\w]', '', header.lower())

def map_columns(header_row, map_keys):
    """헤더 행을 분석하여 표준 컬럼명(예: 'Attention')과 인덱스(위치)를 매핑합니다."""
    col_index_map = {}
    for i, header in enumerate(header_row):
        cleaned = clean_header(header) # 예: 'attention주의력' -> 'attention'
        
        # 맵 키의 모든 변형(예: 'attention', '주의력')에 대해 확인
        for key, standard_name in map_keys.items():
            cleaned_key = clean_header(key) # 예: '주의력' -> '' (이건 문제네)
            # 'lowalpha' 같은 영문 키를 사용해야 함
            if cleaned_key and cleaned_key in cleaned:
                col_index_map[standard_name] = i
                break
    return col_index_map

def parse_comma_separated_string(cell_value):
    """' "1,2,3,..." ' 형식의 문자열을 숫자 리스트로 변환합니다."""
    if pd.isna(cell_value):
        return []
    try:
        # 양쪽의 따옴표나 공백 제거
        val = str(cell_value).strip(' "')
        # 쉼표로 분리하고 숫자로 변환
        return [float(x.strip()) for x in val.split(',') if x.strip()]
    except ValueError:
        return []

def parse_compressed_session_data(df):
    """'압축 세션' 형식의 DataFrame을 파싱합니다. (각 행이 하나의 세션)"""
    sessions = {}
    header_row = df.iloc[0].values
    data_rows = df.iloc[1:].values
    
    col_index_map = map_columns(header_row, COMPRESSED_COLUMN_MAP_KEYS)
    
    if 'Attention' not in col_index_map or 'Delta' not in col_index_map:
        return None # 필수 컬럼이 없으면 압축 형식이 아님

    for index, row in enumerate(data_rows):
        if not row[0]: continue # 빈 행 건너뛰기

        date_val = row[col_index_map.get('Date', 0)] or f"N/A - Session {index + 1}"
        session_id = f"{date_val}_{index}"
        
        session = {
            'id': session_id,
            'Date': date_val,
            'Duration': int(row[col_index_map.get('Duration', 0)] or 0),
            'Tag': row[col_index_map.get('Tag', 0)] or 'None',
            'RawData': {},
            'Format': 'Compressed'
        }
        
        data_length = 0
        all_props = ['Attention', 'Relaxation'] + CHART_ORDER
        
        for prop_name in all_props:
            col_index = col_index_map.get(prop_name)
            data_array = []
            if col_index is not None and col_index < len(row):
                data_array = parse_comma_separated_string(row[col_index])
            
            session['RawData'][prop_name] = data_array
            data_length = max(data_length, len(data_array))
        
        session['SampleCount'] = data_length
        
        # 데이터가 유효한 경우에만 세션 추가
        if data_length > 0 and 'Attention' in session['RawData'] and session['RawData']['Attention']:
            sessions[session_id] = session
            
    return sessions if sessions else None

def parse_standard_time_series_data(df):
    """'표준 시계열' 형식의 DataFrame을 파싱합니다. (각 행이 하나의 시간 지점)"""
    sessions = {}
    header_row = df.iloc[0].values
    data_rows = df.iloc[1:]
    
    col_index_map = map_columns(header_row, STANDARD_COLUMN_MAP_KEYS)

    if 'Attention' not in col_index_map or 'Delta' not in col_index_map:
        return None # 필수 컬럼이 없으면 표준 형식이 아님

    session_id = 'standard_stream_session'
    session = {
        'id': session_id,
        'Date': 'Continuous Stream',
        'Duration': len(data_rows),
        'Tag': 'Standard Time Series',
        'RawData': {},
        'SampleCount': len(data_rows),
        'Format': 'Standard'
    }
    
    all_props = ['Attention', 'Relaxation'] + CHART_ORDER
    
    for prop_name in all_props:
        col_index = col_index_map.get(prop_name)
        if col_index is not None:
            # .loc[행, 열]을 사용하여 데이터 추출 및 숫자 변환
            # pd.to_numeric을 사용하여 숫자가 아닌 값을 NaT/NaN으로 변환 후 0으로 채움
            data_array = pd.to_numeric(data_rows.iloc[:, col_index], errors='coerce').fillna(0).tolist()
            session['RawData'][prop_name] = data_array
        else:
            # 해당 컬럼이 파일에 없으면 0으로 채운 리스트 생성
            session['RawData'][prop_name] = [0] * len(data_rows)
            
    if session['SampleCount'] > 0:
        sessions[session_id] = session
        return sessions
    return None

def calculate_session_metrics(session):
    """세션 데이터를 기반으로 주요 지표를 계산합니다."""
    metrics = {}
    raw_data = session['RawData']
    
    # 1. 파동별 평균 계산
    averages = {}
    for wave in CHART_ORDER:
        if raw_data.get(wave):
            averages[wave] = np.mean(raw_data[wave])
        else:
            averages[wave] = 0
    metrics['WaveAverages'] = averages
    
    # 2. 파생 지수 계산
    ci_num = (averages.get('LowBeta', 0) + averages.get('HighBeta', 0) + 
              averages.get('LowGamma', 0) + averages.get('MidGamma', 0))
    ci_den = (averages.get('Theta', 0) + averages.get('LowAlpha', 0) + averages.get('HighAlpha', 0))
    metrics['ConcentrationIndex'] = (ci_num / ci_den) if ci_den > 0 else (ci_num if ci_num > 0 else 0)
    
    fi_num = averages.get('Theta', 0)
    fi_den = (averages.get('LowBeta', 0) + averages.get('HighBeta', 0))
    metrics['FatigueIndex'] = (fi_num / fi_den) if fi_den > 0 else (fi_num if fi_num > 0 else 0)
    
    at_num = (averages.get('LowAlpha', 0) + averages.get('HighAlpha', 0))
    at_den = averages.get('Theta', 0)
    metrics['AlphaThetaRatio'] = (at_num / at_den) if at_den > 0 else (at_num if at_num > 0 else 0)
    
    # 3. 전체 평균
    metrics['AvgAttention'] = np.mean(raw_data.get('Attention', [0]))
    metrics['AvgRelaxation'] = np.mean(raw_data.get('Relaxation', [0]))
    
    return metrics

def generate_narrative_analysis(session, metrics):
    """계산된 지표를 바탕으로 서술형 리포트를 생성합니다."""
    
    # st.markdown을 사용하여 HTML과 유사한 서식 적용
    st.subheader("3. 상세 뇌파 해석 리포트")
    container = st.container(border=True)

    ci_num = metrics['ConcentrationIndex']
    fi_num = metrics['FatigueIndex']
    
    if metrics['AvgAttention'] >= 60 and ci_num >= 1.2 and fi_num < 0.8:
        overall_state = '매우 높음: 이 세션은 **최적의 집중 상태**와 안정적인 인지 활동을 보여줍니다. 학습 또는 문제 해결에 이상적인 상태였습니다.'
    elif metrics['AvgAttention'] >= 40 and ci_num >= 1.0:
        overall_state = '높음: **양호한 집중력과 적절한 활동 수준**을 유지했습니다. 작업 효율성이 높았을 것으로 보입니다.'
    elif metrics['AvgAttention'] < 40 and fi_num >= 1.0:
        overall_state = '경고: **집중력이 낮고 피로도가 높게** 나타났습니다. 휴식이 필요하거나 주의력 결핍이 발생했을 수 있습니다.'
    elif metrics['AvgRelaxation'] >= 60:
        overall_state = '이완/휴식: **매우 안정되고 이완된 상태**가 지배적이었습니다. 명상, 휴식, 또는 수면 준비에 적합한 상태입니다.'
    else:
        overall_state = '보통: 집중과 이완이 균형을 이루거나, 특정 경향이 두드러지지 않은 평이한 상태입니다.'

    container.markdown(f"**세션 개요** (`{session['Tag']}` / 총 {session['SampleCount']}초)")
    container.markdown(f"""
    - 평균 집중도: **{metrics['AvgAttention']:.1f}**
    - 평균 안정도: **{metrics['AvgRelaxation']:.1f}**
    - 전반적인 뇌 상태: {overall_state}
    """)
    
    container.markdown("---")
    container.markdown(f"**핵심 지수 분석**")
    
    col1, col2, col3 = container.columns(3)
    col1.metric("집중 지수 (CI)", f"{metrics['ConcentrationIndex']:.2f}")
    col2.metric("피로 지수 (FI)", f"{metrics['FatigueIndex']:.2f}")
    col3.metric("알파/세타 비율 (A/T)", f"{metrics['AlphaThetaRatio']:.2f}")

def render_wave_chart(session):
    """시간 경과에 따른 뇌파 차트 (Plotly 이중 Y축)"""
    st.subheader("4. 뇌파 변화 추이")
    
    raw_data = session['RawData']
    time_labels = list(range(1, session['SampleCount'] + 1))
    
    # 이중 Y축을 위한 Plotly Subplots 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Y1 (왼쪽) - 뇌파 파워
    for wave in CHART_ORDER:
        if raw_data.get(wave) and any(raw_data[wave]):
            fig.add_trace(
                go.Scatter(
                    x=time_labels, 
                    y=raw_data[wave],
                    name=WAVE_COLORS[wave]['name'],
                    line=dict(color=WAVE_COLORS[wave]['color'], width=1.5),
                    visible='legendonly' # 기본적으로 숨김
                ),
                secondary_y=False,
            )
            
    # Y2 (오른쪽) - 집중도 및 안정도
    for metric in ['Attention', 'Relaxation']:
        if raw_data.get(metric) and any(raw_data[metric]):
            fig.add_trace(
                go.Scatter(
                    x=time_labels, 
                    y=raw_data[metric],
                    name=WAVE_COLORS[metric]['name'],
                    line=dict(color=WAVE_COLORS[metric]['color'], width=2.5)
                ),
                secondary_y=True,
            )

    # 차트 레이아웃 설정
    fig.update_layout(
        title_text="시간 경과에 따른 뇌파 파워 및 지표 변화",
        xaxis_title="시간 (초 또는 샘플)",
        height=450,
        legend_title_text='지표 선택'
    )
    
    # Y축 제목 설정
    fig.update_yaxes(title_text="뇌파 파워 (μV²)", secondary_y=False)
    fig.update_yaxes(title_text="집중/안정도 (0-100)", secondary_y=True, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)

def render_distribution_chart(metrics):
    """뇌파 주파수별 평균 분포 (Plotly 레이더 차트)"""
    st.subheader("뇌파 주파수별 평균 분포 (로그 스케일 적용)")
    
    wave_averages = metrics['WaveAverages']
    relevant_waves = [wave for wave in CHART_ORDER if wave_averages.get(wave, 0) > 0]
    
    if not relevant_waves:
        st.info("표시할 뇌파 파워 데이터가 없습니다.")
        return

    # 로그 스케일 적용 (JS 로직과 동일)
    log_data = [np.log10(wave_averages[wave] + 1) for wave in relevant_waves]
    
    # 0-100 정규화 (JS 로직과 동일)
    max_val = max(log_data) if log_data else 1
    normalized_data = [(val / max_val) * 100 for val in log_data]
    
    labels = [WAVE_COLORS[wave]['name'].split(' ')[0] for wave in relevant_waves]
    point_colors = [WAVE_COLORS[wave]['groupColor'] for wave in relevant_waves]

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_data,
        theta=labels,
        fill='toself',
        fillcolor='rgba(71, 85, 105, 0.4)',
        line=dict(color='rgb(71, 85, 105)'),
        marker=dict(color=point_colors, size=8) # 마커에 그룹 색상 적용
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False # JS 버전과 동일하게 눈금 숫자 숨김
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_robot_arm_simulation(session):
    """로봇팔 시뮬레이션 (로직만 Python으로 구현)"""
    st.subheader("5. 집중력 기반 로봇팔 시뮬레이션")
    
    max_duration = session['SampleCount']
    default_time = min(60, max_duration)
    
    col1, col2 = st.columns([3, 1])
    
    time_input = col1.number_input(
        "분석할 시간 입력 (초/샘플):", 
        min_value=1, 
        max_value=max_duration, 
        value=default_time,
        step=1
    )
    
    run_button = col2.button("시뮬레이션 실행", use_container_width=True, type="primary")
    
    if run_button:
        time_index = int(time_input) - 1 # 0-based index
        
        if 0 <= time_index < max_duration:
            attention_score = session['RawData']['Attention'][time_index] or 0
            
            if attention_score >= 60:
                st.success(f"**성공!** (시간: {time_input}초, 집중도: {attention_score:.0f})")
                st.markdown("로봇팔이 사물을 **빠르게 정확하게** 집습니다. (최적의 상태)")
            elif attention_score >= 40:
                st.info(f"**성공!** (시간: {time_input}초, 집중도: {attention_score:.0f})")
                st.markdown("로봇팔이 사물을 **적절한 속도로** 집습니다. (양호한 상태)")
            else:
                st.error(f"**실패!** (시간: {time_input}초, 집중도: {attention_score:.0f})")
                st.markdown("**집중력 저하로 사물을 집는 데 실패**했습니다. (주의 필요)")
        else:
            st.warning(f"유효한 시간(1에서 {max_duration} 사이)을 입력해주세요.")

# --- 3. Streamlit 앱 메인 로직 ---

def main():
    st.title("Brainwave Lite 데이터 분석기 (Streamlit Ver.)")
    st.markdown("스프레드시트 파일 (.xlsx, .csv) 업로드를 통해 집중도, 피로도, 안정도를 분석합니다.")

    # 세션 상태 초기화
    if 'all_sessions' not in st.session_state:
        st.session_state.all_sessions = {}
        st.session_state.selected_session_id = None
        st.session_state.format_message = "파일을 업로드하면 형식을 자동 감지하여 분석합니다."

    # 파일 업로더
    uploaded_file = st.file_uploader(
        "1. Brainwave 스프레드시트 파일 (.xlsx, .csv) 선택",
        type=["csv", "xlsx", "xls"]
    )
    
    st.info(st.session_state.format_message)

    if uploaded_file:
        try:
            with st.spinner("파일을 읽고 분석 중입니다..."):
                # 파일 확장자에 따라 다르게 읽기
                if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file, header=None)
                else:
                    # CSV의 경우, 인코딩 문제 방지를 위해 bytes로 읽어 string으로 디코드
                    string_data = uploaded_file.getvalue().decode('utf-8-sig') # BOM 제거
                    df = pd.read_csv(io.StringIO(string_data), header=None, skip_blank_lines=True)
                
                if df.empty:
                    raise ValueError("파일이 비어 있습니다.")

                # 1. 압축 형식 시도
                sessions = parse_compressed_session_data(df)
                if sessions:
                    st.session_state.all_sessions = sessions
                    st.session_state.format_message = f"성공: 압축된 세션 요약 형식 ({len(sessions)}개 세션)으로 분석되었습니다."
                else:
                    # 2. 표준 형식 시도
                    sessions = parse_standard_time_series_data(df)
                    if sessions:
                        st.session_state.all_sessions = sessions
                        st.session_state.format_message = "성공: 표준 시계열 형식 (단일 연속 스트림)으로 분석되었습니다."
                    else:
                        # 두 형식 모두 실패
                        st.session_state.all_sessions = {}
                        st.session_state.format_message = "오류: 업로드된 파일에서 뇌파 데이터를 찾을 수 없거나 파일 형식이 지원되지 않습니다."
                        st.error(st.session_state.format_message)
                
                if st.session_state.all_sessions:
                    st.session_state.selected_session_id = list(st.session_state.all_sessions.keys())[0]

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
            st.session_state.all_sessions = {}

    # --- 분석 결과 표시 (데이터가 로드된 경우) ---
    if st.session_state.all_sessions:
        
        # --- 1. 세션 선택기 (사이드바) ---
        session_ids = list(st.session_state.all_sessions.keys())
        session_options = {
            id: f"[{s['Format']}] {s['Date']} ({s['Tag']})" 
            for id, s in st.session_state.all_sessions.items()
        }

        # st.selectbox의 format_func를 사용하여 옵션 텍스트를 동적으로 표시
        selected_id = st.sidebar.selectbox(
            "세션 선택:",
            options=session_ids,
            format_func=lambda id: session_options[id],
            index=0
        )
        st.session_state.selected_session_id = selected_id
        
        # 선택된 세션 데이터 가져오기
        current_session = st.session_state.all_sessions[selected_id]
        
        # --- 2. 주요 지표 계산 ---
        metrics = calculate_session_metrics(current_session)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("세션 정보")
        st.sidebar.markdown(f"**형식:** {current_session['Format']}")
        st.sidebar.markdown(f"**날짜:** {current_session['Date']}")
        st.sidebar.markdown(f"**태그:** {current_session['Tag']}")
        st.sidebar.markdown(f"**길이:** {current_session['SampleCount']} (초/샘플)")

        # --- 3. 메인 대시보드 ---
        st.header("2. 주요 분석 지표 (평균)")
        
        cols = st.columns(5)
        cols[0].metric("집중도 (Attention)", f"{metrics['AvgAttention']:.1f}")
        cols[1].metric("안정도 (Relaxation)", f"{metrics['AvgRelaxation']:.1f}")
        cols[2].metric("집중 지수 (CI)", f"{metrics['ConcentrationIndex']:.2f}")
        cols[3].metric("피로 지수 (FI)", f"{metrics['FatigueIndex']:.2f}")
        cols[4].metric("알파/세타 (A/T)", f"{metrics['AlphaThetaRatio']:.2f}")

        # 내러티브 리포트
        generate_narrative_analysis(current_session, metrics)

        # 뇌파 추이 차트
        render_wave_chart(current_session)
        
        # 뇌파 분포 차트
        render_distribution_chart(metrics)
        
        # 로봇팔 시뮬레이션
        render_robot_arm_simulation(current_session)

if __name__ == "__main__":
    main()
