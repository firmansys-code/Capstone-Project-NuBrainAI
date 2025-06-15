# 🧠 NuBrain AI 

NuBrain adalah aplikasi klasifikasi tumor otak berbasis deep learning menggunakan TensorFlow, Keras, dan Streamlit. Model dilatih untuk mengenali jenis tumor otak dari citra MRI dengan 4 kategori utama.

---

## Model Pipeline

- Dataset: MRI Brain Tumor Image Dataset (Glioma, Meningioma, Pituitary, Non-Tumor)
- Preprocessing: Image resize (224x224), rescaling, normalization.
- Model: Transfer Learning menggunakan MobileNetV2.
- Export: H5 format + label encoder untuk deployment.
- Deployment: Streamlit multipage web app.

---


## Cara Menggunakan Aplikasi
### Clone Repository
```bash
git clone https://github.com/firmansys-code/Capstone-Project-NuBrainAI.git
cd Capstone-Project-NuBrainAI
```

### Buat Virtual Environment (opsional tapi sangat disarankan)
```bash
conda create --name main-ds python=3.9
conda activate main-ds
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Jalankan Streamlit (Development Mode)
```bash
streamlit run dashboard/app.py
```
