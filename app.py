import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import io # 다운로드 기능을 위해 추가된 부품

# --- 디자인: 페이지 설정 ---
st.set_page_config(layout="wide", page_title="고성능 얼굴 블러 앱")

# --- AI 탐지기 준비 (캐싱) ---
@st.cache_resource
def get_detector():
    return MTCNN()

detector = get_detector()

# --- 블러 처리 함수 ---
def blur_area(image, x, y, w, h):
    face_roi = image[y:y+h, x:x+w]
    if face_roi.size == 0: return image
    
    k = w // 3
    if k % 2 == 0: k += 1
    if k <= 0: k = 1
    blurred_face = cv2.GaussianBlur(face_roi, (k, k), 0)
    image[y:y+h, x:x+w] = blurred_face
    return image

# --- 이미지 처리 함수 ---
def process_image(input_image, min_confidence):
    image = np.array(input_image)
    output_image = image.copy()
    
    results = detector.detect_faces(image)

    count = 0
    if results:
        for result in results:
            confidence = result['confidence']
            if confidence < min_confidence:
                continue
                
            count += 1
            x, y, w, h = result['box']
            x, y = max(0, x), max(0, y)
            blur_area(output_image, x, y, w, h)
                
    return output_image, count

# --- 이미지를 다운로드 가능한 바이트로 변환하는 함수 ---
def convert_image_to_bytes(image_array):
    # OpenCV(numpy) 이미지를 다시 PIL 이미지로 변환
    img = Image.fromarray(image_array)
    # 메모리에 저장할 버퍼 생성
    buf = io.BytesIO()
    # JPEG 형식으로 저장
    img.save(buf, format="JPEG")
    # 바이트 데이터 가져오기
    byte_im = buf.getvalue()
    return byte_im

# --- 화면 디자인 (UI) ---
st.title("🚀 고성능 얼굴 블러 (멀티 업로드 & 다운로드)")
st.write("여러 장의 사진을 한 번에 올리고, 결과물을 다운로드하세요.")

# [사이드바] 설정
st.sidebar.header("⚙️ 설정 패널")
conf_value = st.sidebar.slider("민감도 (낮출수록 더 많이 잡음)", 0.50, 0.99, 0.90, step=0.01)
st.sidebar.info(f"현재 민감도: {conf_value:.2f}")

# [메인] 파일 업로드 (accept_multiple_files=True 로 변경됨!)
uploaded_files = st.file_uploader("이미지 파일을 선택하세요 (여러 개 선택 가능)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# 업로드된 파일이 있으면 반복문으로 하나씩 처리
if uploaded_files:
    st.write(f"총 {len(uploaded_files)}장의 사진을 처리합니다.")
    
    for uploaded_file in uploaded_files:
        # 파일 이름 보여주기용 확장
        with st.expander(f"📷 {uploaded_file.name} 처리 결과 보기", expanded=True):
            
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 화면 나누기
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="원본 사진", use_container_width=True)

            # 변환 실행
            with st.spinner(f"{uploaded_file.name} 변환 중..."):
                processed_image, face_count = process_image(image, conf_value)

            with col2:
                st.image(processed_image, caption=f"변환 결과 ({face_count}명)", use_container_width=True)
                
                # --- 다운로드 버튼 추가 ---
                if face_count > 0:
                    # 이미지를 다운로드용 데이터로 변환
                    byte_img = convert_image_to_bytes(processed_image)
                    
                    btn = st.download_button(
                        label=f"📥 결과 이미지 다운로드 ({uploaded_file.name})",
                        data=byte_img,
                        file_name=f"blurred_{uploaded_file.name}",
                        mime="image/jpeg",
                        key=uploaded_file.name # 버튼마다 고유한 키값 부여
                    )
                else:
                    st.warning("얼굴을 못 찾아서 다운로드 버튼이 없습니다.")
                    
    st.success("모든 작업이 완료되었습니다!")