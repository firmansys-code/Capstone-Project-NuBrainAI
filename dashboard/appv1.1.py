import streamlit as st
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model

# Page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide", page_icon="🧠")

# Hapus query parameter dari URL agar tetap di satu tab
st.query_params.clear()

# Load model and scaler
@st.cache_resource
def load_model():
    with open("knn_brain_model_final.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("knn_scaler_final.pkl", "rb") as f:
        scaler = pickle.load(f)
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    feature_model = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
    return knn, scaler, feature_model

knn, scaler, feature_model = load_model()
labels = ['glioma', 'meningioma', 'no', 'pituitary']

# Top horizontal navigation with stateful session instead of links
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
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: #1f77b46;
            text-align: center;
            padding: 0.5rem 0;
            font-size: 0.9rem;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 NuBrain")

# Sidebar-free button navigation with session state
if 'page' not in st.session_state:
    st.session_state.page = "beranda"

col1, col2, col3, col4 = st.columns([0.5,0.5,0.5,0.5])
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

st.markdown("""---""")

# Halaman: Beranda
if page == "beranda":
    #st.title("🧠 NuBrain")
    st.markdown("""
    ### 📌 Tentang Aplikasi
    **NuBrain** adalah aplikasi berbasis web yang menggunakan model machine learning untuk membantu mendeteksi tumor otak. Aplikasi ini dikembangkan untuk mempermudah pemeriksaan awal secara cepat, informatif, dan mudah digunakan oleh masyarakat umum maupun tenaga medis.

    ### 🧾 Cara Menggunakan
    1. Buka menu **Tes MRI**.
    2. Unggah gambar MRI otak dalam format `.jpg`, `.jpeg`, atau `.png`.
    3. Aplikasi akan menampilkan prediksi jenis tumor berdasarkan gambar yang diunggah.
    4. Hasil prediksi akan ditampilkan di bawah gambar.

    > *Catatan: Aplikasi ini hanya sebagai alat bantu dan bukan pengganti diagnosis dari tenaga medis profesional.*
    
    ### 🎯 Visi
    Memberikan alat bantu diagnosis berbasis AI untuk mendeteksi tumor otak secara dini.

    ### 🎯 Misi
    - Meningkatkan kesadaran masyarakat terhadap tumor otak.
    - Menyediakan tools open-access untuk edukasi dan medis.
    """)

# Halaman: Pengetahuan Umum
elif page == "pengetahuan":
    st.markdown("""
## 🧠 Informasi Lengkap tentang Tumor Otak

Tumor otak adalah pertumbuhan sel abnormal dalam jaringan otak. Bisa bersifat jinak atau ganas. Berikut jenis-jenis tumor otak yang umum:

### 🔬 Glioma
- Berasal dari sel glial (pendukung neuron).
- Subjenis: astrocytoma, oligodendroglioma, glioblastoma.
- [Baca selengkapnya](https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251)

### 🧠 Meningioma
- Berasal dari lapisan meninges.
- Umumnya jinak, tapi bisa agresif.
- [Baca selengkapnya](https://www.hopkinsmedicine.org/health/conditions-and-diseases/meningioma)

### 🩺 Pituitary Tumor
- Tumbuh di kelenjar pituitari yang mengatur hormon.
- Bisa memengaruhi penglihatan atau sistem hormon.
- [Baca selengkapnya](https://my.clevelandclinic.org/health/diseases/17665-pituitary-tumors)

### ✅ No Tumor
- Tidak terdeteksi adanya massa tumor oleh model.
- Perlu dikonfirmasi oleh dokter melalui pemeriksaan lanjut.

📚 Referensi tambahan: [AANS](https://www.aans.org/en/Patients/Neurosurgical-Conditions-and-Treatments/Brain-Tumors)
""")

# Halaman: Tes MRI
elif page == "tes":
    st.title("🧪 Klasifikasi Tumor Otak dari Gambar MRI")
    uploaded = st.file_uploader("Upload gambar MRI", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Gambar yang Diupload", width=300)

        # Preprocessing
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = preprocess_input(img_resized.astype(np.float32))
        img_array = np.expand_dims(img_array, axis=0)

        # Feature extraction and prediction
        features = feature_model.predict(img_array)
        features_scaled = scaler.transform(features)
        pred = knn.predict(features_scaled)[0]

        st.success(f"📌 Prediksi: **{labels[pred].capitalize()} Tumor**")
        st.markdown("""
        > *Catatan: Prediksi ini bersifat informatif dan bukan pengganti diagnosis medis profesional.*
        """)

# Halaman: About
elif page == "about":
    st.title("👨‍💻 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan sebagai bagian dari eksplorasi AI dalam bidang kesehatan.

    - Dibuat menggunakan Python, Streamlit, dan TensorFlow
    - Model: KNN + Ekstraksi Fitur dari EfficientNetB0
    - Dataset: Citra MRI Tumor Otak (Glioma, Meningioma, Pituitary, Non-Tumor)

    Hubungi kami di [LinkedIn](https://www.linkedin.com) atau [GitHub](https://github.com) untuk kolaborasi dan pengembangan lebih lanjut.
    """)


# Sticky Footer
st.markdown("""
<div class="footer">
© 2025 - Aplikasi Deteksi Tumor Otak
</div>
""", unsafe_allow_html=True)
