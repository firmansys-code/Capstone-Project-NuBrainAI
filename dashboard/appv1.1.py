
import streamlit as st
import gdown
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Brain Tumor Classifier", layout="wide", page_icon="🧠")
st.query_params.clear()

@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?id=1kY4qIcs_Jg0Eq4ZpESlMZU5jpd2b9o6U"
    scaler_url = "https://drive.google.com/uc?id=1f6YrFnsntLv-S2gG2RS_zUNwZojDkX-k"
    gdown.download(model_url, "model.pkl", quiet=False)
    gdown.download(scaler_url, "scaler.pkl", quiet=False)
    with open("model.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return knn, scaler

knn, scaler = load_model()
labels = ['glioma', 'meningioma', 'no', 'pituitary']

st.title("🧠 NuBrain")
st.markdown("---")

uploaded = st.file_uploader("Upload gambar MRI", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("L").resize((64, 64))  # grayscale resize
    img_array = np.array(img).flatten().reshape(1, -1).astype(np.float32)
    img_scaled = scaler.transform(img_array)
    prediction = knn.predict(img_scaled)[0]
    st.image(img, caption="Gambar yang Diupload", width=300)
    st.success(f"📌 Prediksi: **{labels[prediction].capitalize()} Tumor**")
    st.markdown("> *Catatan: Prediksi ini bersifat informatif dan bukan pengganti diagnosis medis profesional.*")
