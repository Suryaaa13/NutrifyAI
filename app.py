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
# STYLE (UI/UX RINGAN & NYAMAN)
# =====================================================
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 1.5em;
    }
    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 1.2em;
        margin-bottom: 0.6em;
    }
    .soft-box {
        background-color: #f8f9fa;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 1em;
    }
</style>
""", unsafe_allow_html=True)

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
# FUNGSI: PROSES GAMBAR & DETEKSI YOLO (LABEL AKURAT)
# =====================================================
def process_image(uploaded_file, conf_threshold):
    # Decode image (BGR)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Inferensi YOLO (native)
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

        # 🔥 LABEL DIAMBIL LANGSUNG DARI MODEL
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

# HERO
st.markdown('<div class="main-title">🍽️ NutrifyAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Agen cerdas untuk deteksi makanan dan estimasi nilai gizi '
    'menggunakan YOLO dan Large Language Model.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")
    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.4, 0.05
    )
    st.caption(
        "Nilai confidence lebih tinggi menghasilkan deteksi yang lebih selektif."
    )

# UPLOAD
st.markdown('<div class="section-title">📤 Unggah Gambar Makanan</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Pilih gambar makanan (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    st.markdown('<div class="section-title">🖼️ Pratinjau & Analisis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="soft-box">', unsafe_allow_html=True)
        st.image(uploaded, caption="Gambar Asli", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="soft-box">
        <b>Alur Analisis:</b>
        <ol>
            <li>Deteksi jenis makanan menggunakan YOLO</li>
            <li>Ekstraksi label makanan</li>
            <li>Estimasi nilai gizi menggunakan LLM</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    analyze = st.button(
        "🔎 Analisis Makanan",
        type="primary",
        use_container_width=True
    )

    if analyze:
        with st.spinner("Memproses deteksi dan analisis gizi..."):
            uploaded.seek(0)
            annotated_img, objects, labels = process_image(
                uploaded, conf_threshold
            )

        st.divider()

        if objects:
            st.success(f"Terdeteksi {len(objects)} objek makanan.")

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.image(
                    annotated_img,
                    caption="Hasil Deteksi YOLO",
                    use_container_width=True
                )

            with col2:
                st.markdown('<div class="section-title">🥗 Estimasi Nilai Gizi</div>', unsafe_allow_html=True)
                st.markdown(get_nutrition_info(labels))

            with st.expander("🔍 Detail Teknis Deteksi"):
                st.dataframe(pd.DataFrame(objects))

        else:
            st.warning("Tidak ada makanan yang terdeteksi pada gambar.")
