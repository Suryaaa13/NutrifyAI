import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from groq import Groq

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(
    page_title="NutrifyAI",
    page_icon="🍽️",
    layout="wide"
)

# -----------------------------
# Load Model YOLO (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model YOLO. Pastikan file 'best.pt' ada. Error: {e}")
    st.stop()

# -----------------------------
# Ambil API Key Groq
# -----------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY belum diset di Streamlit Secrets.")
    st.stop()

# -----------------------------
# Inisialisasi Groq Client (FIX UTAMA)
# -----------------------------
client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.1-8b-instant"

# -----------------------------
# Class Labels YOLO
# -----------------------------
CLASS_NAMES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

# -----------------------------
# Helper: LLM Gizi (FIX TOTAL)
# -----------------------------
def get_nutrition_info(makanan_list):
    makanan_unik = sorted(set(makanan_list))
    if not makanan_unik:
        return "Tidak ada makanan terdeteksi."

    makanan_str = ", ".join(makanan_unik)

    prompt = f"""
Kamu adalah seorang ahli gizi profesional.

Berikan ESTIMASI kandungan gizi rata-rata per 1 porsi standar
untuk setiap makanan berikut:
{makanan_str}

ATURAN WAJIB:
- Jawaban HARUS berupa 1 tabel Markdown
- Kolom TEPAT:
| Makanan | Kalori (kkal) | Protein (g) | Lemak (g) | Karbohidrat (g) |
- Gunakan bahasa Indonesia
- TANPA teks penjelasan di luar tabel
"""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah AI ahli gizi. Jawaban hanya berupa tabel Markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ Gagal mengambil data gizi dari AI: {e}"

# -----------------------------
# Helper: Proses Gambar YOLO
# -----------------------------
def process_image(uploaded_file, conf_threshold):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb, conf=conf_threshold)[0]

    detected_objects = []
    makanan_labels = []
    img_draw = image_rgb.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else results.names[cls]
        makanan_labels.append(label)

        detected_objects.append({
            "Nama": label,
            "Confidence": round(conf, 2)
        })

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img_draw,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img_draw, detected_objects, makanan_labels

# -----------------------------
# UI
# -----------------------------
st.title("🍽️ Deteksi Gizi Makanan (YOLO + LLM)")
st.write("Unggah foto makanan untuk mendeteksi dan mengestimasi nilai gizinya.")

with st.sidebar:
    st.header("Pengaturan")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

uploaded = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="Gambar Asli", use_container_width=True)

    if st.button("🔎 Analisis Makanan", type="primary"):
        with st.spinner("Memproses YOLO & AI..."):
            uploaded.seek(0)
            annotated_img, objects, labels = process_image(uploaded, conf_threshold)

            with col2:
                st.image(annotated_img, caption="Hasil Deteksi YOLO", use_container_width=True)

            if objects:
                st.success(f"Terdeteksi {len(objects)} objek makanan.")

                st.subheader("🥗 Analisis Gizi AI")
                st.markdown(get_nutrition_info(labels))

                with st.expander("Lihat Detail Deteksi (JSON)"):
                    st.dataframe(pd.DataFrame(objects))
            else:
                st.warning("Tidak ada makanan terdeteksi.")

