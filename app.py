import streamlit as st
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
from surya.detection import DetectionPredictor
from google import genai

st.set_page_config(page_title="Math OCR (Surya + Gemini)", layout="wide")

def is_inside(inner_bbox, outer_bbox):
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

@st.cache_resource
def load_models(api_key):
    client = genai.Client(api_key=api_key)
    det_predictor = DetectionPredictor()
    return client, det_predictor


st.title("Math Formula Extractor")
st.sidebar.header("Настройки")

api_key = st.sidebar.text_input("Gemini API Key", type="password")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file and api_key:
    client, det_predictor = load_models(api_key)

    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Исходное изображение", width='stretch')

    if st.button("🚀 Начать обработку"):
        with st.spinner("Детекция блоков (Surya)..."):
            predictions = det_predictor([img])
            all_bboxes = predictions[0].bboxes

            valid_boxes = [b for b in all_bboxes if b.confidence >= conf_threshold]
            valid_boxes.sort(key=lambda b: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]), reverse=True)

            final_boxes = []
            for box_to_check in valid_boxes:
                is_nested = False
                for existing_box in final_boxes:
                    if is_inside(box_to_check.bbox, existing_box.bbox):
                        is_nested = True
                        break
                if not is_nested:
                    final_boxes.append(box_to_check)

        st.success(f"Найдено блоков: {len(final_boxes)}")

        draw = ImageDraw.Draw(img)
        results = []

        progress_bar = st.progress(0)

        for idx, box in enumerate(final_boxes):
            coords = box.bbox
            crop_img = img.crop(coords)

            st.write(f"**Обработка блока #{idx}...**")

            try:
                prompt = r"""
                Проанализируй изображение. 
                1. ФИЛЬТРАЦИЯ: Если нет текста/формул, верни пустой ответ.
                2. ФОРМАТ LaTeX: 
                   - Используй \\ для переноса.
                   - Формулы в $...$ или $$...$$.
                   - Текст пиши БЕЗ \text{...}.
                """

                time.sleep(2)

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, crop_img]
                )

                text = response.text.strip() if response.text else ""

                results.append({
                    "id": idx,
                    "latex": text,
                    "confidence": float(box.confidence)
                })

                with st.expander(f"Блок #{idx} - Результат"):
                    c_left, c_right = st.columns([1, 2])
                    c_left.image(crop_img)
                    c_right.code(text, language="latex")
                    if text:
                        st.latex(text)

            except Exception as e:
                st.error(f"Ошибка Gemini на блоке {idx}: {e}")

            draw.rectangle(coords, outline="red", width=3)
            progress_bar.progress((idx + 1) / len(final_boxes))

        st.divider()
        st.subheader("Итоговая разметка")
        st.image(img, width='stretch')

        st.download_button(
            label="Скачать JSON результат",
            data=json.dumps(results, ensure_ascii=False, indent=4),
            file_name="results.json",
            mime="application/json"
        )

elif not api_key:
    st.warning("Пожалуйста, введите API Key в боковой панели.")