import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
import pandas as pd
from ultralytics import YOLO

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(
    page_title="NutrifyAI",
    page_icon="🍽️",
    layout="wide"
)

# -----------------------------
# Load Model & Konfigurasi (Cached)
# -----------------------------
# Menggunakan cache agar model tidak di-load berulang kali setiap ada interaksi
@st.cache_resource
def load_model():
    # Pastikan file best.pt sudah ada di folder yang sama di GitHub
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model YOLO. Pastikan file 'best.pt' ada di root folder. Error: {e}")
    st.stop()

# Ambil API KEY dari Streamlit Secrets (bukan hardcode)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.error("API Key belum diset di Streamlit Secrets!")
    st.stop()

GROQ_MODEL = "llama3-8b-8192" 
# Atau gunakan model lama Anda jika yakin tersedia: "meta-llama/llama-4-maverick-17b-128e-instruct"

CLASS_NAMES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

# -----------------------------
# Helper Functions
# -----------------------------
def get_nutrition_info(makanan_list):
    """
    Request ke Groq LLM untuk estimasi gizi dalam tabel Markdown konsisten
    """
    makanan_unik = sorted(set(makanan_list))
    if not makanan_unik:
        return "Tidak ada makanan terdeteksi."

    makanan_str = ", ".join(makanan_unik)

    prompt = f"""
Kamu adalah seorang ahli gizi profesional.

Berikan ESTIMASI nilai gizi rata-rata per 1 porsi standar
untuk setiap makanan berikut:
{makanan_str}

ATURAN WAJIB:
- Jawaban HARUS berupa 1 tabel Markdown
- Kolom TEPAT sebagai berikut:
| Makanan | Kalori (kkal) | Protein (g) | Lemak (g) | Karbohidrat (g) |
- Gunakan bahasa Indonesia
- Tanpa teks penjelasan di luar tabel
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Kamu adalah AI ahli gizi. "
                    "Jawaban harus berupa tabel Markdown saja."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 700
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Gagal mengambil data gizi dari AI: {e}"
    
def process_image(uploaded_file, conf_threshold):
    # Baca file gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Konversi BGR (OpenCV) ke RGB (untuk diproses model & ditampilkan)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inferensi YOLO
    results = model(image_rgb, conf=conf_threshold)[0]
    
    detected_objects = []
    makanan_labels = []

    # Gambar kotak (Bounding Boxes)
    # Kita gambar di copy image agar aslinya aman
    img_draw = image_rgb.copy()
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Validasi index class (jaga-jaga model custom berbeda index)
        if 0 <= cls < len(CLASS_NAMES):
            label = CLASS_NAMES[cls]
        else:
            label = results.names[cls]
            
        makanan_labels.append(label)
        
        detected_objects.append({
            "Nama": label,
            "Confidence": round(conf, 2)
        })

        # Gambar di Canvas
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_draw, f"{label} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_draw, detected_objects, makanan_labels

# -----------------------------
# UI Layout
# -----------------------------
st.title("🍽️ Deteksi Gizi Makanan (YOLO + LLM)")
st.write("Unggah foto makanan, sistem akan mendeteksi objek dan meminta AI menyusun tabel gizi.")

# Sidebar
with st.sidebar:
    st.header("Pengaturan")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    st.info("Aplikasi ini berjalan full di Cloud tanpa perlu local server Flask.")

# Main Area
uploaded = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded, caption="Gambar Asli", use_container_width=True)
    
    if st.button("🔎 Analisis Makanan", type="primary"):
        with st.spinner("Sedang memproses visi komputer & AI..."):
            # Reset pointer file agar bisa dibaca ulang
            uploaded.seek(0)
            
            # Proses Deteksi
            annotated_img, objects, labels = process_image(uploaded, conf_threshold)
            
            # Tampilkan Gambar Hasil
            with col2:
                st.image(annotated_img, caption="Hasil Deteksi YOLO", use_container_width=True)
            
            # Tampilkan Hasil Text
            if objects:
                st.success(f"Terdeteksi {len(objects)} objek makanan.")
                
                # Request ke LLM
                st.subheader("🥗 Analisis Gizi AI")
                gizi_info = get_nutrition_info(labels)
                st.markdown(gizi_info)
                
                # Tampilkan Data Mentah
                with st.expander("Lihat Detail Deteksi (JSON)"):
                    st.dataframe(pd.DataFrame(objects))
            else:
                st.warning("Tidak ada makanan yang terdeteksi dengan tingkat keyakinan tersebut.")
