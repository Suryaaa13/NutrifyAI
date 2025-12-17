import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from groq import Groq

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="NutrifyAI",
    page_icon="🍽️",
    layout="wide"
)

# =====================================================
# LOAD MODEL YOLO (CACHE)
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model YOLO. Pastikan file 'best.pt' tersedia. Error: {e}")
    st.stop()

# =====================================================
# AMBIL GROQ API KEY
# =====================================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"].strip()
except KeyError:
    st.error("GROQ_API_KEY belum diset di Streamlit Secrets.")
    st.stop()

# =====================================================
# INISIALISASI GROQ CLIENT
# =====================================================
client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

# =====================================================
# FUNGSI: ESTIMASI NILAI GIZI (LLM)
# =====================================================
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

# =====================================================
# FUNGSI: PROSES GAMBAR & DETEKSI YOLO (FIX LABEL)
# =====================================================
def process_image(uploaded_file, conf_threshold):
    # Decode image (SAMA seperti Flask)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Inferensi YOLO (TANPA konversi warna)
    results = model(image)[0]

    detected_objects = []
    makanan_labels = []
    img_draw = image.copy()

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        # 🔥 INI KUNCI UTAMA (SESUAI KODE SEBELUM)
        label = model.names[cls]

        makanan_labels.append(label)
        detected_objects.append({
            "Nama": label,
            "Confidence": round(conf, 2)
        })

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_draw,
            f"{label} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Konversi ke RGB hanya untuk ditampilkan
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    return img_draw, detected_objects, makanan_labels

# =====================================================
# UI APLIKASI
# =====================================================
st.title("🍽️ Deteksi Gizi Makanan (YOLO + LLM)")
st.write("Unggah gambar makanan untuk mendeteksi jenis makanan dan estimasi nilai gizinya.")

with st.sidebar:
    st.header("Pengaturan")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )

uploaded = st.file_uploader(
    "Unggah Gambar Makanan",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="Gambar Asli", use_container_width=True)

    if st.button("🔎 Analisis Makanan", type="primary"):
        with st.spinner("Memproses YOLO & AI..."):
            uploaded.seek(0)

            annotated_img, objects, labels = process_image(
                uploaded,
                conf_threshold
            )

            with col2:
                st.image(
                    annotated_img,
                    caption="Hasil Deteksi YOLO",
                    use_container_width=True
                )

            if objects:
                st.success(f"Terdeteksi {len(objects)} objek makanan.")

                st.subheader("🥗 Analisis Gizi AI")
                st.markdown(get_nutrition_info(labels))

                with st.expander("Lihat Detail Deteksi"):
                    st.dataframe(pd.DataFrame(objects))

            else:
                st.warning("Tidak ada makanan terdeteksi pada gambar.")

