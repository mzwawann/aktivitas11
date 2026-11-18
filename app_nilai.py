import json
import os
import re

import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Langkah 2: Konfigurasi API Gemini (API key TIDAK ditulis di kode)
#   Linux/Mac : export GEMINI_API_KEY="kunci_kamu"
#   Windows   : $env:GEMINI_API_KEY="kunci_kamu"
#   atau .streamlit/secrets.toml -> GEMINI_API_KEY = "kunci_kamu"
# ---------------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if api_key:
    genai.configure(api_key=api_key)

# ---------------------------------------------------------------
# Langkah 3: Judul dan upload file
# ---------------------------------------------------------------
st.title("Analisis Nilai Mahasiswa Menggunakan Gemini AI (JSON Version)")

if not api_key:
    st.error("API key Gemini belum diatur. Set environment variable GEMINI_API_KEY.")
    st.stop()

uploaded_file = st.file_uploader("Upload file CSV data nilai mahasiswa", type=["csv"])

if uploaded_file is not None:
    # Langkah 4: baca dan tampilkan data
    df = pd.read_csv(uploaded_file)
    st.write("Data Nilai Mahasiswa:")
    st.dataframe(df.head())

    # Langkah 5: simpan sementara lalu upload ke Gemini
    temp_path = "uploaded.csv"
    df.to_csv(temp_path, index=False)
    uploaded_to_gemini = genai.upload_file(temp_path, mime_type="text/csv")

    # Langkah 6: prompt
    prompt = """
Analisis data nilai mahasiswa yang telah diupload.
1. Hitung nilai maksimum, minimum, dan rata-rata.
2. Temukan pola menarik dari distribusi nilai.
3. Berikan observasi tentang performa kelas.
4. Berikan rekomendasi untuk meningkatkan performa kelas.
5. Kirimkan hasil analisis dalam format JSON valid.
6. Jangan berikan teks atau penjelasan tambahan di luar JSON.

Contoh format output:
{
"maksimum": 95,
"minimum": 45,
"rata_rata": 72.5,
"pola": "Sebagian besar nilai berada di kisaran 60-80",
"observasi": "Performa kelas cukup baik dengan beberapa siswa berprestasi tinggi",
"rekomendasi": "Fokus pada siswa dengan nilai di bawah rata-rata untuk peningkatan"
}
"""

    # Langkah 7: kirim ke Gemini
    with st.spinner("Sedang mengirim data ke Gemini dan menunggu hasil analisis..."):
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content([prompt, uploaded_to_gemini])

    # Langkah 8: bersihkan respons -> ambil blok { ... }
    raw_text = response.text.strip()
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    json_text = json_match.group(0) if json_match else raw_text

    # Langkah 9-11: parsing, tampilkan, grafik, dan error handling
    try:
        hasil = json.loads(json_text)

        st.subheader("Hasil Analisis Nilai Mahasiswa")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nilai Maksimum", hasil["maksimum"])
        with col2:
            st.metric("Nilai Minimum", hasil["minimum"])
        with col3:
            st.metric("Nilai Rata-rata", hasil["rata_rata"])

        st.write(f"**Pola:** {hasil['pola']}")
        st.write(f"**Observasi:** {hasil['observasi']}")
        st.write(f"**Rekomendasi:** {hasil['rekomendasi']}")

        # Pembanding: hitung langsung dengan pandas untuk verifikasi jawaban AI
        with st.expander("Verifikasi dengan pandas"):
            st.write(
                f"Maks: {df['Nilai'].max()} | Min: {df['Nilai'].min()} | "
                f"Rata-rata: {df['Nilai'].mean():.2f}"
            )

        st.subheader("Grafik Distribusi Nilai Mahasiswa")
        fig, ax = plt.subplots()
        ax.hist(df["Nilai"], bins=10)
        ax.set_xlabel("Rentang Nilai")
        ax.set_ylabel("Jumlah Mahasiswa")
        ax.set_title("Distribusi Nilai Mahasiswa")
        st.pyplot(fig)

    except json.JSONDecodeError:
        st.error("Gagal membaca hasil JSON dari Gemini. Format tidak valid.")
    except KeyError as e:
        st.error(f"Field JSON tidak ditemukan: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
