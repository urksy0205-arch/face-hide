import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import io
import zipfile
from datetime import datetime

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="코끼리공장 사진 모자이크 서비스", initial_sidebar_state="collapsed")

# --- 커스텀 CSS (연한 푸른색 그라데이션 + 중앙 컨테이너) ---
st.markdown("""
<style>
    /* 전체 배경 그라데이션 */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%);
    }
    
    /* 메인 컨테이너 */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        max-width: 1200px;
        margin: 20px auto;
    }
    
    /* 헤더 영역 */
    .header-section {
        display: flex;
        align-items: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 2px solid #e3f2fd;
    }
    
    .logo-title {
        font-size: 28px;
        font-weight: bold;
        color: #1976d2;
        margin-left: 15px;
    }
    
    /* 토글 버튼 스타일 */
    .stRadio > label {
        font-size: 18px;
        font-weight: 600;
        color: #1976d2;
    }
    
    .stRadio > div {
        display: flex;
        gap: 20px;
        background: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 10px 30px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
        border: 2px solid transparent;
    }
    
    .stRadio > div > label:hover {
        border-color: #1976d2;
        transform: translateY(-2px);
    }
    
    /* 다운로드 버튼 */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #1976d2, #2196f3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(25,118,210,0.4);
    }
    
    /* 일괄 다운로드 버튼 */
    .bulk-download {
        text-align: center;
        margin-top: 40px;
        padding-top: 30px;
        border-top: 2px solid #e3f2fd;
    }
    
    /* 슬라이더 */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #1976d2, #2196f3);
    }
    
    /* 파일 업로더 */
    .stFileUploader > div {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed #1976d2;
    }
    
    /* expander */
    .streamlit-expanderHeader {
        background: #e3f2fd;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- AI 탐지기 준비 (캐싱) ---
@st.cache_resource
def get_detector():
    return MTCNN()

detector = get_detector()

# --- 모자이크 처리 함수 ---
def mosaic_area(image, x, y, w, h, ratio=0.05):
    """모자이크 효과"""
    face_roi = image[y:y+h, x:x+w]
    if face_roi.size == 0: 
        return image
    
    # 이미지 축소 후 확대로 모자이크 효과
    small_h = max(1, int(h * ratio))
    small_w = max(1, int(w * ratio))
    
    temp = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    image[y:y+h, x:x+w] = mosaic_face
    return image

# --- 이미지 처리 함수 ---
def process_image(input_image, min_confidence, is_auto_mode):
    """이미지 처리 (자동/수동 모드)"""
    image = np.array(input_image)
    output_image = image.copy()
    
    results = detector.detect_faces(image)

    count = 0
    if results:
        for result in results:
            confidence = result['confidence']
            
            # 자동 모드는 민감도 무시하고 모두 처리
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
    """여러 이미지를 ZIP으로 묶기"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, img_bytes in processed_images_data:
            zip_file.writestr(filename, img_bytes)
    return zip_buffer.getvalue()

# ==================== UI 시작 ====================

# 헤더 (로고 + 제목)
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logo.png", width=120)  # 로고 이미지
with col_title:
    st.markdown('<div class="logo-title">사진 모자이크 서비스</div>', unsafe_allow_html=True)

st.markdown("---")

# === 모드 선택 (토글) ===
mode = st.radio(
    "처리 모드 선택",
    ["🤖 자동 모드 (AI가 최대한 많이 탐지)", "⚙️ 수동 모드 (민감도 직접 조절)"],
    horizontal=True
)

is_auto_mode = "자동" in mode

# === 설정 영역 ===
if is_auto_mode:
    st.info("🤖 **자동 모드**: AI가 가장 강력한 민감도로 얼굴을 최대한 많이 찾아 모자이크 처리합니다.")
    conf_value = 0.50  # 자동 모드는 최대 민감도
else:
    st.info("⚙️ **수동 모드**: 슬라이더로 민감도를 조절할 수 있습니다. (낮을수록 더 많이 탐지)")
    conf_value = st.slider("민감도 조절", 0.50, 0.99, 0.90, step=0.01)
    st.caption(f"현재 민감도: {conf_value:.2f}")

st.markdown("---")

# === 파일 업로드 ===
uploaded_files = st.file_uploader(
    "📤 이미지 파일을 선택하세요 (여러 개 선택 가능)", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

# === 처리 결과 저장용 리스트 ===
processed_images_data = []

# === 업로드된 파일 처리 ===
if uploaded_files:
    st.success(f"✅ 총 {len(uploaded_files)}장의 사진을 처리합니다.")
    
    for idx, uploaded_file in enumerate(uploaded_files, 1):
        with st.expander(f"📷 [{idx}] {uploaded_file.name}", expanded=True):
            
            # 이미지 로드
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 좌우 배치
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="🖼️ 원본 사진", use_container_width=True)

            # 처리 실행
            with st.spinner(f"🔄 {uploaded_file.name} 처리 중..."):
                processed_image, face_count = process_image(image, conf_value, is_auto_mode)

            with col2:
                st.image(processed_image, caption=f"✨ 모자이크 결과 ({face_count}개 얼굴)", use_container_width=True)
                
                # 개별 다운로드 버튼
                byte_img = convert_image_to_bytes(processed_image)
                processed_images_data.append((f"mosaic_{uploaded_file.name}", byte_img))
                
                st.download_button(
                    label=f"💾 이 이미지 다운로드",
                    data=byte_img,
                    file_name=f"mosaic_{uploaded_file.name}",
                    mime="image/jpeg",
                    key=f"download_{idx}"
                )
    
    # === 일괄 다운로드 버튼 ===
    if len(processed_images_data) > 1:
        st.markdown('<div class="bulk-download">', unsafe_allow_html=True)
        st.markdown("### 📦 모든 결과 한번에 다운로드")
        
        zip_data = create_zip(processed_images_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label=f"📥 전체 다운로드 ({len(processed_images_data)}장) - ZIP",
            data=zip_data,
            file_name=f"코끼리공장_모자이크_{timestamp}.zip",
            mime="application/zip",
            key="bulk_download"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.balloons()
    st.success("🎉 모든 작업이 완료되었습니다!")
