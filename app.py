import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import io
import zipfile
from datetime import datetime

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="코끼리공장 모자이크 도우미", initial_sidebar_state="collapsed")

# --- 커스텀 CSS ---
st.markdown("""
<style>
    /* 전체 배경 - 더 연한 푸른색 그라데이션 */
    .stApp {
        background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 50%, #d6ebff 100%);
    }
    
    /* 메인 컨테이너 */
    .main-container {
        background: white;
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        max-width: 1400px;
        margin: 20px auto;
    }
    
    /* 헤더 영역 */
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 40px;
        padding-bottom: 25px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .service-title {
        font-size: 32px;
        font-weight: 700;
        color: #212121;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* 섹션 제목 */
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 15px;
        margin-top: 30px;
    }
    
    /* 라디오 버튼 영역 */
    .stRadio > div {
        display: flex;
        gap: 15px;
        background: transparent;
        padding: 0;
        margin-bottom: 20px;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 12px 24px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        border: 2px solid #e0e0e0;
        font-size: 15px;
        color: #424242;
        font-weight: 500;
    }
    
    .stRadio > div > label:hover {
        border-color: #1976d2;
        background: #f5f5f5;
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color: #1976d2;
        background: #e3f2fd;
        color: #1976d2;
    }
    
    /* 정보 박스 */
    .stAlert {
        background: #e3f2fd;
        border-left: 4px solid #1976d2;
        border-radius: 8px;
        padding: 16px;
        margin: 20px 0;
    }
    
    /* 슬라이더 영역 */
    .slider-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
    }
    
    /* 파일 업로더 */
    .stFileUploader > div {
        background: #fafafa;
        border-radius: 12px;
        padding: 30px;
        border: 2px dashed #bdbdbd;
        text-align: center;
    }
    
    .stFileUploader label {
        font-size: 15px;
        font-weight: 600;
        color: #424242;
    }
    
    /* expander */
    .streamlit-expanderHeader {
        background: #f5f5f5;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px;
    }
    
    /* 다운로드 버튼 */
    .stDownloadButton > button {
        background: #1976d2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background: #1565c0;
        box-shadow: 0 4px 12px rgba(25,118,210,0.3);
    }
    
    /* 일괄 다운로드 영역 */
    .bulk-download {
        text-align: center;
        margin-top: 50px;
        padding-top: 40px;
        border-top: 1px solid #e0e0e0;
    }
    
    .bulk-download h3 {
        font-size: 20px;
        font-weight: 700;
        color: #212121;
        margin-bottom: 20px;
    }
    
    /* 이미지 크기 제한 */
    .stImage {
        max-height: 400px !important;
    }
    
    .stImage img {
        max-height: 400px !important;
        object-fit: contain;
    }
    
    /* 구분선 */
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 30px 0;
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
st.markdown('<div class="header-container">', unsafe_allow_html=True)
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.image("logo.png", width=100)
with col_title:
    st.markdown('<div class="service-title">모자이크 도우미</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# === 모드 선택 섹션 ===
st.markdown('<div class="section-title">처리 모드 선택</div>', unsafe_allow_html=True)

mode = st.radio(
    "",
    ["🤖 자동 모드", "⚙️ 수동 모드"],
    horizontal=True,
    label_visibility="collapsed"
)

is_auto_mode = "자동" in mode

st.markdown("---")

# ==================== 자동 모드 ====================
if is_auto_mode:
    st.info("🤖 **자동 모드**: AI가 가장 강력한 민감도로 얼굴을 최대한 많이 찾아 모자이크 처리합니다.")
    conf_value = 0.50
    
    # 파일 업로드
    st.markdown('<div class="section-title">📤 이미지 파일을 선택하세요 (여러 개 선택 가능)</div>', unsafe_allow_html=True)
    
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
                    st.image(image, caption="🖼️ 원본 사진", use_column_width=True)

                with st.spinner(f"🔄 {uploaded_file.name} 처리 중..."):
                    processed_image, face_count = process_image(image, conf_value, is_auto_mode)

                with col2:
                    st.image(processed_image, caption=f"✨ 모자이크 결과 ({face_count}개 얼굴)", use_column_width=True)
                    
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
            st.markdown('<div class="bulk-download">', unsafe_allow_html=True)
            st.markdown('<h3>📦 모든 결과 한번에 다운로드</h3>', unsafe_allow_html=True)
            
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
    
    # 좌우 레이아웃: 왼쪽 슬라이더, 오른쪽 이미지
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.markdown('<div class="slider-container">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ 민감도 조절")
        conf_value = st.slider(
            "민감도", 
            0.50, 0.99, 0.90, 
            step=0.01,
            help="낮을수록 더 많은 얼굴을 탐지합니다"
        )
        st.caption(f"현재 민감도: **{conf_value:.2f}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 파일 업로드
        st.markdown("#### 📤 파일 업로드")
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
            
            # 원본과 결과를 좌우로 배치
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**🖼️ 원본 사진**")
                st.image(image, use_column_width=True)
            
            with img_col2:
                st.markdown("**✨ 모자이크 결과**")
                with st.spinner("처리 중..."):
                    processed_image, face_count = process_image(image, conf_value, False)
                st.image(processed_image, use_column_width=True)
                st.caption(f"탐지된 얼굴: **{face_count}개**")
            
            # 다운로드 버튼
            st.markdown("---")
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
