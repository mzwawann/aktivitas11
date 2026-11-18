import json
import os
import re

import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------
# Langkah 1: Konfigurasi API Key Gemini
# JANGAN menulis API key langsung di kode. Gunakan environment variable:
#   Windows (PowerShell): $env:GEMINI_API_KEY="kunci_kamu"
#   Linux/Mac           : export GEMINI_API_KEY="kunci_kamu"
# atau simpan di .streamlit/secrets.toml:  GEMINI_API_KEY = "kunci_kamu"
# ---------------------------------------------------------------
def load_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    return key


GEMINI_API_KEY = load_api_key()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------
# Langkah 2: Ambil data cuaca dari Open-Meteo
# ---------------------------------------------------------------
def get_weather_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max",
        "timezone": "Asia/Jakarta",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(
        {
            "Tanggal": data["daily"]["time"],
            "Suhu Maksimum (°C)": data["daily"]["temperature_2m_max"],
        }
    )
    return df


# ---------------------------------------------------------------
# Langkah 4: Kirim data ke Gemini
# ---------------------------------------------------------------
def analyze_weather_with_gemini(df):
    # DataFrame diubah menjadi teks agar bisa dimasukkan ke prompt
    text_data = df.to_string(index=False)

    prompt = f"""
Berikut data suhu maksimum harian selama periode pengamatan:
{text_data}

Buat analisis tren suhu dan prediksi kondisi suhu 7 hari ke depan.
Jawab dalam format JSON dengan field:
- tren (naik/turun/stabil)
- prediksi (ringkasan singkat)
- alasan (penjelasan)
Jangan berikan teks di luar JSON.
"""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


def extract_json(raw_text):
    """Ambil blok { ... } dari respons Gemini lalu ubah menjadi dict."""
    match = re.search(r"\{.*\}", raw_text.strip(), re.DOTALL)
    return json.loads(match.group(0) if match else raw_text)


# ---------------------------------------------------------------
# Langkah 3: Antarmuka Streamlit
# ---------------------------------------------------------------
st.title("Prediksi Tren Cuaca dengan Gemini AI")
st.markdown(
    "Aplikasi ini mengambil data cuaca dari [Open-Meteo](https://open-meteo.com) "
    "dan mengirimkannya ke **Gemini AI** untuk mendapatkan analisis tren dan prediksi suhu."
)

city = st.text_input("Kota", "Yogyakarta")
lat = st.number_input("Latitude", value=-7.7972, format="%.4f")
lon = st.number_input("Longitude", value=110.3688, format="%.4f")
start_date = st.date_input("Tanggal Mulai", pd.to_datetime("2025-09-24"))
end_date = st.date_input("Tanggal Akhir", pd.to_datetime("2025-10-24"))

if st.button("Ambil dan Analisis Data"):
    if not GEMINI_API_KEY:
        st.error("API key Gemini belum diatur. Set environment variable GEMINI_API_KEY.")
        st.stop()
    if start_date > end_date:
        st.error("Tanggal mulai tidak boleh setelah tanggal akhir.")
        st.stop()

    try:
        with st.spinner("Mengambil data cuaca..."):
            df = get_weather_data(lat, lon, start_date, end_date)
        st.success("Data berhasil diambil!")
        st.dataframe(df)

        # Visualisasi data
        st.subheader("Grafik Suhu Harian")
        fig, ax = plt.subplots()
        ax.plot(df["Tanggal"], df["Suhu Maksimum (°C)"], marker="o")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Suhu Maksimum (°C)")
        ax.set_title(f"Pergerakan Suhu Harian di {city}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        # Langkah 5: kirim ke Gemini dan tampilkan hasil
        with st.spinner("Mengirim data ke Gemini AI..."):
            result = analyze_weather_with_gemini(df)

        st.subheader("Hasil Analisis AI")
        try:
            hasil = extract_json(result)
            st.write(f"**Tren:** {hasil.get('tren', '-')}")
            st.write(f"**Prediksi:** {hasil.get('prediksi', '-')}")
            st.write(f"**Alasan:** {hasil.get('alasan', '-')}")
            with st.expander("Lihat JSON mentah"):
                st.json(hasil)
        except json.JSONDecodeError:
            st.warning("Respons bukan JSON valid, ditampilkan apa adanya:")
            st.write(result)

    except requests.RequestException as e:
        st.error(f"Gagal mengambil data cuaca: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
