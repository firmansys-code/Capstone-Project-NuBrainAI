import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# Page config
st.set_page_config(page_title="MRI Brain Tumor Classifier", layout="wide", page_icon="🧠")

# Load model and label encoder (cached)
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("dashboard/end_to_end_model.h5")
    with open("dashboard/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model_and_encoder()
class_names = le.classes_

# Simple CSS styling for menu
st.markdown("""
    <style>
        .mainmenu {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 2rem;
        }
        .mainmenu button {
            background: none;
            border: none;
            color: #333;
            font-size: 1.1rem;
            cursor: pointer;
            padding: 0.3rem 1rem;
            border-radius: 5px;
            transition: background 0.3s, color 0.3s;
        }
        .mainmenu button:hover {
            background-color: #1f77b4;
            color: white;
        }
        .mainmenu button.active {
            background-color: #1f77b4;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation state
if 'page' not in st.session_state:
    st.session_state.page = "beranda"

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Beranda"):
        st.session_state.page = "beranda"
with col2:
    if st.button("📚 Pengetahuan"):
        st.session_state.page = "pengetahuan"
with col3:
    if st.button("🧪 Tes MRI"):
        st.session_state.page = "tes"
with col4:
    if st.button("👨‍💻 About"):
        st.session_state.page = "about"

page = st.session_state.page

# Halaman: Beranda
if page == "beranda":
    st.markdown('<h1 style="text-align: center;">🧠 NuBrain Tumor</h1>', unsafe_allow_html=True)
    st.write("""
    ### 📌 Tentang Aplikasi
    **NuBrain** adalah aplikasi berbasis web yang menggunakan model AI untuk membantu mendeteksi tumor otak melalui citra MRI. Aplikasi ini dikembangkan untuk mempermudah pemeriksaan awal secara cepat, informatif, dan mudah digunakan oleh masyarakat umum maupun tenaga medis.

    ### 🧾 Cara Menggunakan
    1. Buka menu **Tes MRI**.
    2. Unggah gambar MRI otak dalam format `.jpg`, `.jpeg`, atau `.png`.
    3. Aplikasi akan menampilkan prediksi jenis tumor berdasarkan gambar yang diunggah.
    4. Hasil prediksi akan ditampilkan di bawah gambar.

    > *Catatan: Aplikasi ini hanya sebagai alat bantu dan bukan pengganti diagnosis dari tenaga medis profesional.*

    ### 🎯 Visi
    Memberikan alat bantu diagnosis berbasis AI untuk mendeteksi tumor otak.

    ### 🚀 Misi
    - Meningkatkan kesadaran masyarakat terhadap tumor otak.
    - Menyediakan tools open-access untuk edukasi dan medis.""")

# Halaman: Pengetahuan Umum
elif page == "pengetahuan":
    st.title("📚 Pengetahuan Tentang Tumor Otak")
    st.write("""
        Tumor otak merupakan pertumbuhan jaringan sel abnormal. 
        Kategori klasifikasi dalam model ini:
        - **Glioma**: Tumor jaringan glial otak.
        - **Meningioma**: Tumor di jaringan meninges.
        - **Pituitary Tumor**: Tumor di kelenjar pituitari.
        - **Non-Tumor**: Tidak ada indikasi tumor.
        
        Deteksi dini penting untuk menentukan keberhasilan pengobatan.
        Model AI ini membantu klasifikasi awal berbasis citra MRI.
        
        > ⚠ Hasil prediksi ini hanya sebagai alat bantu edukasi, bukan pengganti konsultasi dokter.
    """)

# Halaman: Tes MRI
elif page == "tes":
    st.title("🧪 Tes Klasifikasi Gambar MRI")

    uploaded = st.file_uploader("Upload Gambar MRI", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Gambar MRI yang Diunggah", width=300)

        # Preprocessing sesuai training
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        if st.button("Prediksi"):
            prediction = model.predict(image_array)
            predicted_class = class_names[np.argmax(prediction)]
            st.success(f"Prediksi: {predicted_class}")

# Halaman: About
elif page == "about":
    st.title("👨‍💻 Tentang Aplikasi")
    st.write("""
        Aplikasi ini dikembangkan menggunakan TensorFlow, Streamlit, dan MobileNetV2 untuk klasifikasi tumor otak.
        
        Model deep learning melakukan ekstraksi fitur dari gambar MRI untuk mendeteksi 4 kategori tumor otak.
        Pipeline melibatkan:
        - Data preprocessing & augmentasi
        - Transfer Learning dengan MobileNetV2
        - Model deployment dengan Streamlit
        
        > 📢 Aplikasi ini untuk keperluan edukasi AI medis.
    """)

# Sticky Footer
st.markdown("""
<div class="footer" style="position:fixed;bottom:0;width:100%;text-align:center;color:gray;">
© 2025 - NuBrain
</div>
""", unsafe_allow_html=True)
