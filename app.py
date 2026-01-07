import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import io
import zipfile
from datetime import datetime

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="코끼리공장 | 모자이크 도우미", initial_sidebar_state="collapsed")

# --- 커스텀 CSS ---
st.markdown("""
<style>
    /* 전체 배경 - 연한 그라데이션 하늘색 */
    .stApp {
        background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    /* Streamlit 기본 패딩 제거 */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    
    /* 메인 컨테이너 - 투명한 흰색 박스 */
    .main-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 50px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 0 auto;
    }
    
    /* 헤더 - 로고와 타이틀 */
    .header-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 50px;
    }
    
    .header-logo {
        font-size: 28px;
        font-weight: 700;
        color: #1976d2;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    
    .header-divider {
        color: #bdbdbd;
        font-size: 28px;
        font-weight: 300;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #212121;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    
    /* 섹션 타이틀 */
    .section-label {
        font-size: 15px;
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 12px;
        margin-top: 30px;
    }
    
    /* 라디오 버튼 스타일 */
    .stRadio > div {
        display: flex;
        gap: 12px;
        margin-bottom: 25px;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 14px 28px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 15px;
        font-weight: 500;
        color: #424242;
    }
    
    .stRadio > div > label:hover {
        border-color: #1976d2;
        background: #f5f5f5;
    }
    
    /* Info 박스 */
    .stAlert {
        background: #e3f2fd;
        border: none;
        border-left: 4px solid #1976d2;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 20px 0;
    }
    
    /* 파일 업로더 */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #bdbdbd;
        border-radius: 12px;
        padding: 30px;
    }
    
    [data-testid="stFileUploader"] label {
        font-size: 15px;
        font-weight: 600;
        color: #424242;
    }
    
    /* 이미지 크기 제한 */
    [data-testid="stImage"] img {
        max-height: 350px;
        object-fit: contain;
        border-radius: 8px;
    }
    
    /* 다운로드 버튼 */
    .stDownloadButton > button {
        background: #1976d2;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stDownloadButton > button:hover {
        background: #1565c0;
        box-shadow: 0 4px 16px rgba(25, 118, 210, 0.3);
        transform: translateY(-1px);
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    [data-testid="stExpanderToggleIcon"] {
        color: #1976d2;
    }
    
    /* 슬라이더 */
    .stSlider {
        padding: 15px 0;
    }
    
    /* Success/Warning 메시지 */
    .stSuccess {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 12px 16px;
    }
    
    /* 일괄 다운로드 영역 */
    .bulk-section {
        margin-top: 40px;
        padding-top: 30px;
        border-top: 2px solid #e0e0e0;
        text-align: center;
    }
    
    .bulk-title {
        font-size: 18px;
        font-weight: 700;
        color: #212121;
        margin-bottom: 20px;
    }
    
    /* 구분선 */
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 35px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- AI 탐지기 준비 ---
@st.cache_resource
def get_detector():
    return MTCNN()

detector = get_detector()

# --- 모자이크 처리 함수 ---
def mosaic_area(image, x, y, w, h, ratio=0.05):
    face_roi = image[y:y+h, x:x+w]
    if face_roi.size == 0: 
        return image
    
    small_h = max(1, int(h * ratio))
    small_w = max(1, int(w * ratio))
    
    temp = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    image[y:y+h, x:x+w] = mosaic_face
    return image

# --- 이미지 처리 함수 ---
def process_image(input_image, min_confidence, is_auto_mode):
    image = np.array(input_image)
    output_image = image.copy()
    
    results = detector.detect_faces(image)

    count = 0
    if results:
        for result in results:
            confidence = result['confidence']
            
            if not is_auto_mode and confidence < min_confidence:
                continue
                
            count += 1
            x, y, w, h = result['box']
            x, y = max(0, x), max(0, y)
            mosaic_area(output_image, x, y, w, h)
                
    return output_image, count

# --- 이미지를 바이트로 변환 ---
def convert_image_to_bytes(image_array):
    img = Image.fromarray(image_array)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# --- ZIP 파일 생성 ---
def create_zip(processed_images_data):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, img_bytes in processed_images_data:
            zip_file.writestr(filename, img_bytes)
    return zip_buffer.getvalue()

# ==================== UI 시작 ====================

# 메인 카드 시작
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# 헤더
col_logo, col_space = st.columns([3, 1])
with col_logo:
    header_col1, header_col2, header_col3 = st.columns([0.15, 0.05, 0.8])
    with header_col1:
        st.image("logo.png", width=90)
    with header_col2:
        st.markdown('<div class="header-divider">|</div>', unsafe_allow_html=True)
    with header_col3:
        st.markdown('<div class="header-title">모자이크 도우미</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 모드 선택
st.markdown('<div class="section-label">처리 모드 선택</div>', unsafe_allow_html=True)

mode = st.radio(
    "",
    ["🤖 자동 모드", "⚙️ 수동 모드"],
    horizontal=True,
    label_visibility="collapsed"
)

is_auto_mode = "자동" in mode

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== 자동 모드 ====================
if is_auto_mode:
    st.info("🤖 **자동 모드**: AI가 가장 강력한 민감도로 얼굴을 최대한 많이 찾아 모자이크 처리합니다.")
    conf_value = 0.50
    
    st.markdown('<div class="section-label">📤 이미지 파일을 선택하세요 (여러 개 선택 가능)</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "파일 선택", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="auto_uploader"
    )
    
    processed_images_data = []
    
    if uploaded_files:
        st.success(f"✅ 총 {len(uploaded_files)}장의 사진을 처리합니다.")
        
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            with st.expander(f"📷 [{idx}] {uploaded_file.name}", expanded=False):
                
                image = Image.open(uploaded_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🖼️ 원본 사진**")
                    st.image(image, use_column_width=True)

                with st.spinner(f"🔄 처리 중..."):
                    processed_image, face_count = process_image(image, conf_value, is_auto_mode)

                with col2:
                    st.markdown(f"**✨ 모자이크 결과 ({face_count}개 얼굴)**")
                    st.image(processed_image, use_column_width=True)
                    
                    byte_img = convert_image_to_bytes(processed_image)
                    processed_images_data.append((f"mosaic_{uploaded_file.name}", byte_img))
                    
                    st.download_button(
                        label=f"💾 이 이미지 다운로드",
                        data=byte_img,
                        file_name=f"mosaic_{uploaded_file.name}",
                        mime="image/jpeg",
                        key=f"download_auto_{idx}"
                    )
        
        # 일괄 다운로드
        if len(processed_images_data) > 1:
            st.markdown('<div class="bulk-section">', unsafe_allow_html=True)
            st.markdown('<div class="bulk-title">📦 모든 결과 한번에 다운로드</div>', unsafe_allow_html=True)
            
            zip_data = create_zip(processed_images_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            col_center = st.columns([1, 2, 1])[1]
            with col_center:
                st.download_button(
                    label=f"📥 전체 다운로드 ({len(processed_images_data)}장) - ZIP",
                    data=zip_data,
                    file_name=f"코끼리공장_모자이크_{timestamp}.zip",
                    mime="application/zip",
                    key="bulk_download_auto"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("🎉 모든 작업이 완료되었습니다!")

# ==================== 수동 모드 ====================
else:
    st.info("⚙️ **수동 모드**: 슬라이더로 민감도를 조절하면 실시간으로 결과를 확인할 수 있습니다.")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown('<div class="section-label">⚙️ 민감도 조절</div>', unsafe_allow_html=True)
        conf_value = st.slider(
            "민감도", 
            0.50, 0.99, 0.90, 
            step=0.01,
            help="낮을수록 더 많은 얼굴을 탐지합니다",
            label_visibility="collapsed"
        )
        st.caption(f"현재 민감도: **{conf_value:.2f}**")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-label">📤 파일 업로드</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "이미지 선택", 
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed",
            key="manual_uploader"
        )
    
    with col_right:
        if uploaded_file:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**🖼️ 원본 사진**")
                st.image(image, use_column_width=True)
            
            with img_col2:
                with st.spinner("처리 중..."):
                    processed_image, face_count = process_image(image, conf_value, False)
                st.markdown(f"**✨ 모자이크 결과**")
                st.image(processed_image, use_column_width=True)
                st.caption(f"탐지된 얼굴: **{face_count}개**")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            byte_img = convert_image_to_bytes(processed_image)
            
            col_download = st.columns([1, 2, 1])[1]
            with col_download:
                st.download_button(
                    label=f"💾 결과 이미지 다운로드",
                    data=byte_img,
                    file_name=f"mosaic_{uploaded_file.name}",
                    mime="image/jpeg",
                    key="download_manual"
                )
        else:
            st.info("👈 왼쪽에서 이미지를 업로드하고 민감도를 조절해보세요!")

# 메인 카드 끝
st.markdown('</div>', unsafe_allow_html=True)
