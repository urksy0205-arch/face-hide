import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN # 새로운 AI 모델 가져오기

# --- 디자인: 페이지 설정 ---
st.set_page_config(layout="wide", page_title="고성능 얼굴 블러 앱")

# --- (중요) 새로운 AI 탐지기 준비 (한 번만 실행됨) ---
# 이 줄이 실행될 때 시간이 조금 걸릴 수 있습니다.
@st.cache_resource
def get_detector():
    return MTCNN()

detector = get_detector()

# --- 블러 처리 함수 (이건 기존과 동일) ---
def blur_area(image, x, y, w, h):
    face_roi = image[y:y+h, x:x+w]
    # 얼굴 영역이 이미지 범위를 벗어나지 않게 안전장치
    if face_roi.size == 0: return image
    
    k = w // 3
    if k % 2 == 0: k += 1
    if k <= 0: k = 1
    blurred_face = cv2.GaussianBlur(face_roi, (k, k), 0)
    image[y:y+h, x:x+w] = blurred_face
    return image

# --- 핵심: 이미지를 처리하는 함수 (MTCNN 사용) ---
def process_image(input_image, min_confidence):
    # 이미지 준비 (PIL -> numpy 배열로 변환)
    image = np.array(input_image)
    # MTCNN은 RGB 이미지를 사용하므로 BGR로 변환할 필요 없음
    output_image = image.copy()
    
    # --- 새로운 AI로 얼굴 찾기! ---
    # detect_faces 함수가 얼굴 위치와 확률을 다 찾아줍니다.
    results = detector.detect_faces(image)

    count = 0 # 잡은 얼굴 개수 세기
    if results:
        for result in results:
            # 확신도(confidence)가 사용자가 설정한 값보다 높을 때만 처리
            confidence = result['confidence']
            if confidence < min_confidence:
                continue
                
            count += 1
            # MTCNN이 주는 좌표 정보 가져오기
            x, y, w, h = result['box']
            # 가끔 좌표가 음수가 나올 때를 대비한 안전장치
            x, y = max(0, x), max(0, y)
            
            # 블러 처리
            blur_area(output_image, x, y, w, h)
                
    return output_image, count

# --- 화면 디자인 (UI) ---
st.title("🚀 고성능 얼굴 블러 (MTCNN 적용)")
st.write("더 강력한 AI로 멀리 있는 작은 얼굴까지 찾아냅니다.")

# [왼쪽 사이드바]
st.sidebar.header("⚙️ 설정 패널")
# 민감도 슬라이더 (MTCNN은 확신도를 0.0~1.0 사이로 줍니다)
conf_value = st.sidebar.slider("민감도 (낮출수록 더 많이 잡음)", 0.50, 0.99, 0.90, step=0.01)
st.sidebar.info(f"현재 민감도: {conf_value:.2f} (이 값보다 확신이 높아야 얼굴로 인정)")

# 메인 화면
uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # RGB 모드가 아니면 변환 (가끔 흑백이나 투명 배경 이미지 대비)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 화면 나누기
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본 사진")
        st.image(image, use_container_width=True)

    # 변환 실행
    with st.spinner("고성능 AI가 얼굴을 정밀 탐색 중... (조금 더 걸릴 수 있어요)"):
        processed_image, face_count = process_image(image, conf_value)

    with col2:
        st.subheader(f"변환 결과 ({face_count}명 감지됨)")
        st.image(processed_image, use_container_width=True)
        
    if face_count == 0:
        st.warning("얼굴을 못 찾았어요. 왼쪽 민감도를 조금 낮춰보세요!")
    else:
        st.success(f"와우! 총 {face_count}명의 얼굴을 찾아 가렸습니다! 🎉")