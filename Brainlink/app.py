import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import re
from openpyxl import load_workbook # .xlsx 파일 읽기 지원

# --- 1. 상수 및 설정 (Constants and Configuration) ---

# 뇌파 주파수 및 색상 설정
WAVE_CONFIG = {
    'Delta': {'name': "델타 (깊은 수면)", 'color': 'rgb(239, 68, 68)', 'group_color': 'rgb(239, 68, 68)'},
    'Theta': {'name': "세타 (졸음/이완)", 'color': 'rgb(59, 130, 246)', 'group_color': 'rgb(59, 130, 246)'},
    'LowAlpha': {'name': "저주파 알파 (휴식)", 'color': 'rgb(245, 158, 11)', 'group_color': 'rgb(16, 185, 129)'},
    'HighAlpha': {'name': "고주파 알파 (평온)", 'color': 'rgb(16, 185, 129)', 'group_color': 'rgb(16, 185, 129)'},
    'LowBeta': {'name': "저주파 베타 (집중/인지)", 'color': 'rgb(139, 92, 246)', 'group_color': 'rgb(139, 92, 246)'},
    'HighBeta': {'name': "고주파 베타 (스트레스)", 'color': 'rgb(249, 115, 22)', 'group_color': 'rgb(139, 92, 246)'},
    'LowGamma': {'name': "저주파 감마 (고차인지)", 'color': 'rgb(107, 114, 128)', 'group_color': 'rgb(31, 41, 55)'},
    'MidGamma': {'name': "고주파 감마 (통합)", 'color': 'rgb(31, 41, 55)', 'group_color': 'rgb(31, 41, 55)'},
    'Attention': {'name': "집중도", 'color': 'rgb(22, 163, 74)'},
    'Relaxation': {'name': "안정도", 'color': 'rgb(147, 51, 234)'}
}

CHART_ORDER = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MidGamma']

# 다국어/다중 형식 헤더 매핑을 위한 정규식
COMPRESSED_COLUMN_MAP = {
    r'(attention|注意力|집중도)': 'Attention',
    r'(relaxation|放松度|안정도|이완)': 'Relaxation',
    r'(delta|δ波|델타)': 'Delta',
    r'(theta|θ波|세타)': 'Theta',
    r'(low[\s-]?alpha|低α波|저주파[\s-]?알파)': 'LowAlpha',
    r'(high[\s-]?alpha|高α波|고주파[\s-]?알파)': 'HighAlpha',
    r'(low[\s-]?beta|低β波|저주파[\s-]?베타)': 'LowBeta',
    r'(high[\s-]?beta|高β波|고주파[\s-]?베타)': 'HighBeta',
    r'(low[\s-]?gamma|低γ波|저주파[\s-]?감마)': 'LowGamma',
    r'(mid[\s-]?gamma|高γ波|고주파[\s-]?감마)': 'MidGamma',
    r'(date|日期|날짜|날자)': 'Date',
    r'(duration|时长|초|샘플)': 'Duration',
    r'(tag|备注|태그)': 'Tag',
    r'(time|시간|timestamp|타임스탬프)': 'Time'
}


# --- 2. 데이터 파싱 함수 (Data Parsing Functions) ---

def clean_header(header):
    """헤더 텍스트를 정리하여 매핑에 사용할 수 있도록 소문자화하고 공백을 제거합니다."""
    return str(header).strip().lower()

def map_columns(header_row, col_map):
    """주어진 헤더 행을 정규식 맵을 사용하여 표준화된 키에 매핑합니다."""
    col_index_map = {}
    
    for i, header in enumerate(header_row):
        cleaned_header = clean_header(header)
        for pattern, standard_key in col_map.items():
            if re.search(pattern, cleaned_header):
                col_index_map[standard_key] = i
                break
    return col_index_map

def parse_compressed_session_data(df, col_map):
    """
    압축된 형식(한 행이 하나의 세션 요약)을 처리합니다.
    세션 데이터(예: '48,48,35,...')는 문자열로 압축되어 있습니다.
    """
    sessions = {}
    
    # 필수 컬럼 확인
    essential_cols = ['Attention', 'Relaxation', 'Delta']
    if not all(k in col_map for k in essential_cols):
        return {}

    for index, row in df.iterrows():
        try:
            date_val = row[col_map.get('Date')] if col_map.get('Date') is not None else f'N/A - Session {index + 1}'
            session_id = f"{date_val}_{index}"
            
            session = {
                'id': session_id,
                'Date': date_val,
                'Tag': row[col_map.get('Tag')] if col_map.get('Tag') is not None else 'None',
                'RawData': {},
                'Format': 'Compressed',
                'SampleCount': 0
            }
            
            data_length = 0
            all_props = ['Attention', 'Relaxation'] + CHART_ORDER
            
            for prop_name in all_props:
                col_index = col_map.get(prop_name)
                if col_index is not None:
                    cell_value = str(row[col_index]).strip()
                    
                    # 따옴표 제거
                    if cell_value.startswith('"') and cell_value.endswith('"'):
                        cell_value = cell_value[1:-1]

                    # 쉼표로 분리하여 숫자 배열로 변환
                    data_array = [float(s.strip()) for s in cell_value.split(',') if s.strip().replace('.', '', 1).isdigit()]
                    session['RawData'][prop_name] = data_array
                    
                    if data_array and len(data_array) > data_length:
                        data_length = len(data_array)
                else:
                    session['RawData'][prop_name] = [] # 컬럼이 없는 경우 빈 배열로 초기화

            session['SampleCount'] = data_length
            
            # 모든 RawData 배열의 길이를 가장 긴 배열에 맞게 조정 (분석 일관성을 위해)
            for prop_name in all_props:
                if len(session['RawData'][prop_name]) < data_length:
                    session['RawData'][prop_name].extend([0] * (data_length - len(session['RawData'][prop_name])))

            if data_length > 0 and session['RawData']['Attention']:
                sessions[session_id] = session
                
        except Exception as e:
            st.warning(f"세션 {index + 1} 처리 중 오류 발생: {e}")
            continue
            
    return sessions

def parse_standard_timeseries_data(df, col_map):
    """
    표준 시계열 형식(한 행이 하나의 타임 포인트)을 처리합니다.
    단일 연속 스트림 세션으로 집계됩니다.
    """
    sessions = {}
    
    # 필수 컬럼 확인
    essential_cols = ['Attention', 'Relaxation', 'Delta', 'Theta']
    if not all(k in col_map for k in essential_cols):
        return {}
        
    session_id = 'standard_stream_session'
    session_data = {
        'id': session_id,
        'Date': '연속 스트림',
        'Tag': '표준 시계열',
        'RawData': {},
        'Format': 'Standard',
        'SampleCount': len(df)
    }
    
    all_props = ['Attention', 'Relaxation'] + CHART_ORDER
    
    for prop_name in all_props:
        col_index = col_map.get(prop_name)
        if col_index is not None:
            # 데이터프레임에서 직접 열 추출
            # .iloc[:, col_index]를 사용하는 대신 컬럼 이름을 사용하기 위해 헤더를 임시로 변경해야 함.
            # 하지만 안전하게 하기 위해, 추출 시점에 컬럼 이름을 표준화했다고 가정합니다.
            
            # 임시 컬럼 이름 생성 (파싱 시점에 헤더가 표준화되지 않았을 수 있으므로)
            try:
                # df의 컬럼 이름을 표준화된 키로 변경
                df.rename(columns={df.columns[col_index]: prop_name}, inplace=True)
                session_data['RawData'][prop_name] = df[prop_name].astype(float).replace([np.inf, -np.inf, np.nan], 0).tolist()
            except Exception as e:
                # 표준화된 키가 없으면 해당 열을 0으로 채움
                session_data['RawData'][prop_name] = [0] * len(df)
                st.warning(f"'{prop_name}' 컬럼 처리 중 오류 발생: {e}. 0으로 채웁니다.")

    if session_data['RawData'].get('Attention') and len(session_data['RawData']['Attention']) > 0:
         sessions[session_id] = session_data
         
    return sessions

@st.cache_data
def load_and_parse_data(uploaded_file):
    """업로드된 파일을 읽고 두 가지 형식으로 파싱을 시도합니다."""
    
    # 파일 확장자에 따라 데이터 로드
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['xlsx', 'xls']:
        try:
            df = pd.read_excel(uploaded_file, header=None)
        except Exception as e:
            return None, f"XLSX 파일 로드 오류: {e}"
    elif file_extension == 'csv':
        try:
            # CSV는 인코딩 문제에 취약하므로, 여러 인코딩을 시도해봅니다.
            # Streamlit은 파일 객체를 전달하므로, io.StringIO를 사용합니다.
            uploaded_file.seek(0)
            data = uploaded_file.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(data), header=None)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            data = uploaded_file.read().decode("euc-kr") # 한국어 인코딩 시도
            df = pd.read_csv(io.StringIO(data), header=None)
        except Exception as e:
            return None, f"CSV 파일 로드 오류: {e}"
    else:
        return None, "지원되지 않는 파일 형식입니다. (.csv, .xlsx만 지원)"

    # 데이터프레임의 첫 번째 행을 헤더로 사용
    header_row = df.iloc[0].tolist()
    data_df = df.iloc[1:].reset_index(drop=True)
    
    # --- 압축 형식 시도 ---
    # 압축 형식의 컬럼 매핑은 데이터 행(data_df)에 적용
    col_map_compressed = map_columns(header_row, COMPRESSED_COLUMN_MAP)
    compressed_sessions = parse_compressed_session_data(data_df, col_map_compressed)

    if compressed_sessions:
        return compressed_sessions, f"성공: 압축된 세션 요약 형식 ({len(compressed_sessions)}개 세션)으로 분석되었습니다."

    # --- 표준 형식 시도 ---
    # 표준 형식의 컬럼 매핑은 데이터 행(data_df)에 적용
    col_map_standard = map_columns(header_row, COMPRESSED_COLUMN_MAP) # 동일한 맵 사용 (시계열 데이터에서 Time 열만 제외)
    
    # 표준 형식 처리를 위해 데이터프레임의 헤더를 매핑 결과로 표준화 (임시 복사본 사용)
    temp_df = df.copy()
    temp_df.columns = header_row # 첫 행을 헤더로 설정
    
    # 컬럼 이름을 표준화된 키로 변경
    standard_df = temp_df.iloc[1:].reset_index(drop=True)
    standard_df.columns = [COMPRESSED_COLUMN_MAP.get(k, k) for k in header_row]

    standard_sessions = parse_standard_timeseries_data(standard_df, col_map_standard)
    
    if standard_sessions:
        return standard_sessions, f"성공: 표준 시계열 형식 (단일 연속 스트림)으로 분석되었습니다."

    return None, "데이터에서 유효한 뇌파 정보(Attention, Relaxation, Delta, Theta 등)를 찾을 수 없거나 형식이 지원되지 않습니다."


# --- 3. 분석 및 계산 함수 (Analysis and Calculation) ---

def calculate_metrics(session):
    """세션의 핵심 지표를 계산합니다."""
    raw_data = session['RawData']
    
    # 1. Calculate Averages
    averages = {}
    for wave in CHART_ORDER:
        data = np.array(raw_data.get(wave, [0]))
        averages[wave] = np.mean(data[data != 0]) if len(data[data != 0]) > 0 else 0
        
    # 2. Calculate Derived Indices
    
    # Total Active (Beta + Gamma)
    active_sum = (averages['LowBeta'] + averages['HighBeta'] + 
                  averages['LowGamma'] + averages['MidGamma'])
    
    # Total Relaxed (Theta + Alpha)
    relaxed_sum = (averages['Theta'] + averages['LowAlpha'] + averages['HighAlpha'])
    
    # Concentration Index (CI) = Active / Relaxed
    ci = active_sum / relaxed_sum if relaxed_sum > 0 else 0
    
    # Fatigue Index (FI) = Theta / Beta
    beta_sum = averages['LowBeta'] + averages['HighBeta']
    fi = averages['Theta'] / beta_sum if beta_sum > 0 else 0
    
    # Alpha/Theta Ratio (A/T Ratio) = Alpha / Theta
    alpha_sum = averages['LowAlpha'] + averages['HighAlpha']
    at_ratio = alpha_sum / averages['Theta'] if averages['Theta'] > 0 else 0
    
    # Overall Scores
    avg_attention = np.mean(raw_data.get('Attention', [0]))
    avg_relaxation = np.mean(raw_data.get('Relaxation', [0]))

    session.update({
        'AvgAttention': avg_attention,
        'AvgRelaxation': avg_relaxation,
        'ConcentrationIndex': round(ci, 2),
        'FatigueIndex': round(fi, 2),
        'AlphaThetaRatio': round(at_ratio, 2),
        'WaveAverages': averages
    })
    
    return session

def generate_narrative_analysis(session):
    """계산된 지표를 기반으로 서술형 리포트를 생성합니다."""
    
    ci = session['ConcentrationIndex']
    fi = session['FatigueIndex']
    at_ratio = session['AlphaThetaRatio']
    avg_attention = session['AvgAttention']
    avg_relaxation = session['AvgRelaxation']
    wave_averages = session['WaveAverages']
    
    # 1. Overall State
    overall_state = ''
    if avg_attention >= 60 and ci >= 1.2 and fi < 0.8:
        overall_state = '**최적의 집중 상태**와 안정적인 인지 활동을 보여줍니다. 학습 또는 문제 해결에 이상적인 상태였습니다.'
    elif avg_attention >= 40 and ci >= 1.0:
        overall_state = '**양호한 집중력과 적절한 활동 수준**을 유지했습니다. 작업 효율성이 높았을 것으로 보입니다.'
    elif avg_attention < 40 and fi >= 1.0:
        overall_state = '**집중력이 낮고 피로도가 높게** 나타났습니다. 휴식이 필요하거나 주의력 결핍이 발생했을 수 있습니다.'
    elif avg_relaxation >= 60:
        overall_state = '**매우 안정되고 이완된 상태**가 지배적이었습니다. 명상, 휴식, 또는 수면 준비에 적합한 상태입니다.'
    else:
        overall_state = '집중과 이완이 균형을 이루거나, 특정 경향이 두드러지지 않은 평이한 상태입니다.'

    # 2. Dominant Wave Analysis
    relevant_waves = {k: v for k, v in wave_averages.items() if v > 0}
    if relevant_waves:
        dominant_wave = max(relevant_waves, key=relevant_waves.get)
        wave_name = WAVE_CONFIG[dominant_wave]['name']
        
        if 'Beta' in dominant_wave or 'Gamma' in dominant_wave:
            wave_summary = f"주요 주파수는 **{wave_name}**로, 활발한 사고, 집중력, 또는 스트레스 수준의 증가를 시사합니다."
        elif 'Theta' in dominant_wave or 'Delta' in dominant_wave:
            wave_summary = f"주요 주파수는 **{wave_name}**로, 깊은 이완, 졸음, 또는 인지 부하 상태를 나타냅니다."
        elif 'Alpha' in dominant_wave:
            wave_summary = f"주요 주파수는 **{wave_name}**로, 고요하고 명상적인 휴식 상태가 잘 유지되었음을 보여줍니다."
    else:
        wave_summary = '뇌파 파워 데이터가 존재하지 않아 상세 주파수 분석이 제한됩니다.'
        
    # 3. Ratio Interpretation
    ci_comment = '낮음 (활동력 저하)'
    if ci > 1.2: ci_comment = '높음 (뇌가 효율적으로 작동)'
    elif ci > 0.8: ci_comment = '보통 (균형 잡힌 인지 활동)'

    fi_comment = '낮음 (경계 상태 유지)'
    if fi > 1.0: fi_comment = '높음 (피로하거나 주의 산만)'
    elif fi > 0.8: fi_comment = '보통 (적절한 수준)'
    
    at_comment = '낮음 (과도한 각성 또는 산만)'
    if at_ratio > 2.0: at_comment = '높음 (이완된 집중, 명상 상태에 유리)'
    elif at_ratio > 1.0: at_comment = '보통 (안정적인 이완 상태)'

    narrative = f"""
    ### 🧠 세션 개요
    * **측정 정보:** 날짜 '{session['Date']}', 태그 '{session['Tag']}', 총 {session['SampleCount']} {session['Format'] == 'Standard' and '샘플' or '초'}.
    * **평균 점수:** 집중도 **{avg_attention:.1f}점**, 안정도 **{avg_relaxation:.1f}점**.
    * **전반적인 상태:** {overall_state}

    ### 📊 핵심 지수 분석
    | 지수 | 값 | 해석 |
    | :--- | :--- | :--- |
    | **집중 지수 (CI)** | **{ci:.2f}** | {ci_comment} |
    | **피로 지수 (FI)** | **{fi:.2f}** | {fi_comment} |
    | **알파/세타 비율 (A/T Ratio)** | **{at_ratio:.2f}** | {at_comment} |
    
    > **참고:** CI (인지 활동/휴식), FI (졸음/활동), A/T Ratio (이완된 집중).

    ### ✨ 주파수 분포 해석
    * {wave_summary}
    """
    
    return narrative

# --- 4. 시각화 함수 (Visualization Functions) ---

def create_wave_line_chart(session):
    """시간 경과에 따른 뇌파 파워 및 점수 변화를 Plotly 차트로 생성합니다."""
    
    df_raw = pd.DataFrame(session['RawData'])
    if df_raw.empty:
        return go.Figure()

    fig = go.Figure()
    
    # 뇌파 데이터 (주축)
    for wave in CHART_ORDER:
        if wave in df_raw.columns and not df_raw[wave].isnull().all():
            fig.add_trace(go.Scatter(
                y=df_raw[wave],
                mode='lines',
                name=WAVE_CONFIG[wave]['name'],
                line=dict(color=WAVE_CONFIG[wave]['color'], width=1.5),
                opacity=0.6,
                yaxis='y1' # 뇌파 파워 축
            ))

    # 집중도/안정도 데이터 (보조축)
    scores = ['Attention', 'Relaxation']
    for score in scores:
         if score in df_raw.columns and not df_raw[score].isnull().all():
            fig.add_trace(go.Scatter(
                y=df_raw[score],
                mode='lines',
                name=WAVE_CONFIG[score]['name'] + ' (점수)',
                line=dict(color=WAVE_CONFIG[score]['color'], width=2.5),
                yaxis='y2' # 점수 축
            ))

    # 레이아웃 설정
    fig.update_layout(
        title_text="시간 경과에 따른 뇌파 파워 및 집중/안정도 변화",
        height=500,
        xaxis=dict(title='시간 (샘플/초)', rangeslider=dict(visible=True)),
        yaxis1=dict(
            title='뇌파 파워 (μV²)',
            side='left',
            rangemode='tozero',
            showgrid=True,
            titlefont=dict(color="black")
        ),
        yaxis2=dict(
            title='집중/안정도 점수 (0-100)',
            overlaying='y1',
            side='right',
            range=[0, 100],
            showgrid=False,
            titlefont=dict(color=WAVE_CONFIG['Attention']['color'])
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_distribution_radar_chart(session):
    """평균 뇌파 파워 분포를 Plotly 레이더 차트로 생성합니다."""
    
    wave_averages = session['WaveAverages']
    relevant_waves = [w for w in CHART_ORDER if wave_averages[w] > 0]
    
    if not relevant_waves:
        st.info("뇌파 파워 데이터가 부족하여 분포 차트를 생성할 수 없습니다.")
        return go.Figure()
        
    # 로그 스케일 변환 (log10(x+1) 사용)
    data = [np.log10(wave_averages[w] + 1) for w in relevant_waves]
    labels = [WAVE_CONFIG[w]['name'].split(' ')[0] for w in relevant_waves]
    
    # 최대값에 대해 정규화 (레이더 차트 시각화를 위해)
    max_val = max(data) if data else 1
    normalized_data = [(d / max_val) * 100 for d in data]

    # 포인트 색상을 그룹 색상으로 설정
    point_colors = [WAVE_CONFIG[w]['group_color'] for w in relevant_waves]

    fig = go.Figure(data=[
        go.Scatterpolar(
            r=normalized_data,
            theta=labels,
            fill='toself',
            name='평균 뇌파 파워',
            line_color='rgb(71, 85, 105)',
            fillcolor='rgba(71, 85, 105, 0.4)',
            marker=dict(
                size=10,
                color=point_colors,
                line=dict(color='white', width=1)
            )
        )
    ])

    fig.update_layout(
        title_text="뇌파 주파수별 평균 분포 (로그 스케일)",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False # 눈금 숨기기
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        height=450
    )
    
    return fig


# --- 5. Streamlit UI 구성 (Streamlit UI Composition) ---

def main():
    st.set_page_config(layout="wide", page_title="Brainwave Analyzer")

    st.title("🧠 Brainwave Lite 데이터 분석기")
    st.markdown("스프레드시트 파일 (.xlsx, .csv) 업로드를 통해 집중도, 피로도, 안정도를 분석합니다. ")
    
    st.markdown("---")
    
    # 1. 파일 업로드 섹션
    uploaded_file = st.file_uploader(
        "1. Brainwave 스프레드시트 파일 (.xlsx, .csv) 선택",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        sessions, message = load_and_parse_data(uploaded_file)
        st.info(message)
        
        if sessions:
            session_ids = list(sessions.keys())
            
            # 세션 선택 섹션
            session_options = {
                id: f"[{sessions[id]['Tag']}] {sessions[id]['Date']} ({sessions[id]['SampleCount']} {'샘플' if sessions[id]['Format'] == 'Standard' else '초'})"
                for id in session_ids
            }
            
            selected_session_id = st.selectbox(
                "2. 분석할 세션을 선택하세요:",
                options=session_ids,
                format_func=lambda x: session_options[x]
            )
            
            # 현재 세션 데이터 로드 및 분석
            current_session = sessions[selected_session_id]
            calculated_session = calculate_metrics(current_session)
            
            st.markdown("---")

            # 3. 주요 분석 지표
            st.header("3. 주요 분석 지표")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="집중도 (Attention Score)", 
                    value=f"{calculated_session['AvgAttention']:.1f}", 
                    help="Brainlink 원시 집중 점수의 평균 (0-100)"
                )
            with col2:
                st.metric(
                    label="안정도 (Relaxation Score)", 
                    value=f"{calculated_session['AvgRelaxation']:.1f}", 
                    help="Brainlink 원시 이완 점수의 평균 (0-100)"
                )
            with col3:
                st.metric(
                    label="파생 집중 지수 (CI)", 
                    value=f"{calculated_session['ConcentrationIndex']:.2f}",
                    help="인지 활동 / 휴식 (높을수록 집중)"
                )
            with col4:
                st.metric(
                    label="파생 피로 지수 (FI)", 
                    value=f"{calculated_session['FatigueIndex']:.2f}",
                    help="졸음 / 활동 (높을수록 피로)"
                )
            with col5:
                st.metric(
                    label="알파/세타 비율 (A/T Ratio)", 
                    value=f"{calculated_session['AlphaThetaRatio']:.2f}",
                    help="이완된 집중/명상 상태 지표 (Alpha / Theta)"
                )

            st.markdown("---")
            
            # 4. 상세 뇌파 해석 리포트
            st.header("4. 상세 뇌파 해석 리포트")
            narrative = generate_narrative_analysis(calculated_session)
            st.markdown(narrative)
            
            st.markdown("---")
            
            # 5. 뇌파 변화 추이 시각화
            st.header("5. 뇌파 변화 추이")
            
            col_chart, col_radar = st.columns([3, 2])
            
            with col_chart:
                st.subheader("시간 경과에 따른 뇌파 파워 변화")
                line_chart = create_wave_line_chart(calculated_session)
                st.plotly_chart(line_chart, use_container_width=True)
                
            with col_radar:
                st.subheader("뇌파 주파수별 평균 분포")
                radar_chart = create_distribution_radar_chart(calculated_session)
                st.plotly_chart(radar_chart, use_container_width=True)

            st.markdown("---")

            # 6. 집중력 기반 시뮬레이션 (간소화)
            st.header("6. 집중력 기반 시뮬레이션 (간소화)")
            st.markdown("집중도 점수에 따라 로봇팔이 사물을 잡는 성공 여부를 시뮬레이션합니다.")
            
            max_duration = calculated_session['SampleCount']
            
            # 시뮬레이션 시간 선택 (1초/샘플 단위)
            simulation_time = st.slider(
                "분석할 시간 선택 (샘플/초):",
                min_value=1,
                max_value=max_duration,
                value=min(60, max_duration),
                step=1,
                help=f"세션 시작 후 몇 초/샘플 시점의 집중도를 분석할지 선택하세요."
            )
            
            time_index = simulation_time - 1
            sim_attention_score = calculated_session['RawData']['Attention'][time_index]
            
            sim_col1, sim_col2 = st.columns([1, 2])
            
            with sim_col1:
                st.metric(
                    label=f"{simulation_time}초/샘플 시점의 집중도",
                    value=f"{sim_attention_score:.1f}점",
                    delta_color="off"
                )

            with sim_col2:
                if sim_attention_score >= 60:
                    st.success(f"**로봇팔 작동 성공!** 🤖✨\n\n집중도 {sim_attention_score:.1f}점은 **최적의 인지 상태**를 나타내며, 로봇팔은 사물을 **빠르고 정확하게** 집습니다.")
                    st.image("https://placehold.co/300x100/10b981/ffffff?text=SUCCESS", caption="집중력 높음: 정확한 작업")
                elif sim_attention_score >= 40:
                    st.warning(f"**로봇팔 작동 보통.** 🤖\n\n집중도 {sim_attention_score:.1f}점은 **양호한 상태**이지만, 약간의 지연이나 불안정성이 있을 수 있습니다.")
                    st.image("https://placehold.co/300x100/f59e0b/ffffff?text=MODERATE+SUCCESS", caption="집중력 보통: 적절한 작업")
                else:
                    st.error(f"**로봇팔 작동 실패.** 🚫\n\n집중도 {sim_attention_score:.1f}점은 **인지 활동 저하** 상태를 나타내며, 로봇팔은 사물을 집는 데 실패했습니다. 휴식이 필요합니다.")
                    st.image("https://placehold.co/300x100/ef4444/ffffff?text=FAILURE", caption="집중력 낮음: 작업 실패")

        else:
            st.error("파일에서 분석할 수 있는 유효한 세션 데이터를 찾을 수 없습니다. 파일 형식을 확인해주세요.")
            st.code("필수 데이터 헤더: Attention, Relaxation, Delta, Theta 등 (영어, 한국어, 중국어 지원)", language='markdown')


if __name__ == '__main__':
    main()
