# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jelaskan latar belakang bisnis dari perushaan tersebut.

### Permasalahan Bisnis

Tuliskan seluruh permasalahan bisnis yang akan diselesaikan.

### Cakupan Proyek

Tuliskan cakupan proyek yang akan dikerjakan.

### Persiapan

Sumber data: ....

Setup environment:

```

```

## Business Dashboard

Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

## Menjalankan Sistem Machine Learning

Aplikasi ini menggunakan model machine learning (XGBoost) untuk memprediksi risiko seorang mahasiswa mengalami dropout dari perkuliahan. Model dilatih menggunakan dataset yang berisi informasi akademik, demografis, dan sosial-ekonomi mahasiswa.

### Cara Menjalankan Aplikasi Secara Lokal

#### Prasyarat

- Python 3.7 atau lebih baru
- pip (Python package manager)

#### Langkah-langkah Instalasi (Windows)

**Metode 1: Menggunakan file batch**

1. Clone repository atau download source code aplikasi
2. Double-click pada file `run_app.bat`
3. File batch akan otomatis menginstal dependency dan menjalankan aplikasi

**Metode 2: Menggunakan command line**

1. Clone repository atau download source code aplikasi

2. Buka terminal/command prompt dan navigasi ke direktori proyek:

   ```
   cd path/to/project
   ```

3. Install semua dependensi yang diperlukan:

   ```
   pip install -r requirements.txt
   ```

4. Jalankan setup script untuk memverifikasi instalasi:

   ```
   python setup.py
   ```

5. Jalankan aplikasi Streamlit:

   ```
   streamlit run app.py
   ```

6. Aplikasi akan terbuka secara otomatis di browser web Anda. Jika tidak, buka browser dan kunjungi URL berikut:
   ```
   http://localhost:8501
   ```

#### Mengatasi Error "ModuleNotFoundError: No module named 'imblearn'"

Jika Anda mengalami error "ModuleNotFoundError: No module named 'imblearn'", pastikan untuk menginstal library imbalanced-learn:

```
pip install imbalanced-learn
```

Kemudian restart aplikasi Streamlit.

### Fitur Utama Aplikasi

- Prediksi probabilitas dropout mahasiswa
- Visualisasi faktor-faktor yang mempengaruhi dropout
- Rekomendasi tindakan berdasarkan tingkat risiko
- Analisis statistik dataset

### Mengakses Versi Online

Aplikasi ini juga tersedia secara online melalui Streamlit Community Cloud di:
[https://dropout-prediction.streamlit.app](https://dropout-prediction.streamlit.app)

### Deployment ke Streamlit Community Cloud

Untuk men-deploy aplikasi ini ke Streamlit Community Cloud:

1. Buat repository GitHub baru dan push semua file proyek ke dalamnya

2. Pastikan struktur repository Anda memiliki:

   - app.py (file utama aplikasi)
   - requirements.txt (daftar dependensi)
   - model_dropout.pkl (model machine learning yang telah dilatih)
   - feature_names.json (nama fitur yang digunakan model)
   - data.csv (dataset untuk analisis)

3. Kunjungi [share.streamlit.io](https://share.streamlit.io/) dan login dengan akun GitHub Anda

4. Klik "New app" dan pilih repository yang berisi proyek ini

5. Konfigurasi deployment:

   - Repository: [nama-repo]
   - Branch: main
   - Main file path: app.py
   - Klik "Deploy!"

6. Tunggu beberapa saat hingga deployment selesai dan aplikasi siap digunakan

## Conclusion

Jelaskan konklusi dari proyek yang dikerjakan.

### Rekomendasi Action Items

Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.

- action item 1
- action item 2
