<!-- filepath: d:\IDCamp\Mahir\Last\README.md -->
# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Perusahaan Edutech ini adalah institusi pendidikan tinggi yang menyediakan berbagai program studi untuk mahasiswa. Namun, institusi ini menghadapi tingkat putus sekolah (dropout) yang cukup tinggi. Hal ini tidak hanya memengaruhi reputasi institusi tetapi juga menyebabkan kerugian finansial dan menurunkan efektivitas program pendidikan yang ditawarkan.

### Permasalahan Bisnis

1. Tingkat dropout mahasiswa yang tinggi dan tidak terprediksi dengan baik
2. Kesulitan dalam mengidentifikasi mahasiswa yang berisiko putus sekolah sejak dini
3. Kurangnya pemahaman tentang faktor-faktor utama yang berkontribusi pada keputusan mahasiswa untuk berhenti studi
4. Tidak adanya sistem peringatan dini untuk intervensi tepat waktu
5. Penurunan retensi mahasiswa yang berdampak pada keuangan institusi

### Cakupan Proyek

1. Menganalisis data historis mahasiswa untuk memahami pola dan tren dropout
2. Membangun model machine learning untuk memprediksi risiko dropout mahasiswa
3. Mengidentifikasi faktor-faktor kunci yang memengaruhi keputusan dropout
4. Mengembangkan sistem prediksi berbasis web yang dapat digunakan oleh staf akademik
5. Memberikan rekomendasi intervensi berdasarkan analisis untuk meningkatkan retensi mahasiswa

### Persiapan

Sumber data: Dataset akademik mahasiswa yang berisi informasi demografis, kinerja akademik, dan status kelulusan.

Setup environment:

1. Pastikan Python sudah terinstall

```bash
python --version
```

2. Buat virtual environment (jika belum ada)

```bash
python -m venv .env
```

3. Aktifkan virtual environment (pilih sesuai sistem operasi)

Untuk Windows PowerShell:

```bash
.\.env\Scripts\Activate.ps1
```

Untuk Command Prompt Windows:

```bash
.\.env\Scripts\activate.bat
```

Untuk bash/Linux/MacOS:

```bash
source .env/bin/activate
```

4. Install dependencies yang diperlukan

```bash
pip install -r requirements.txt
```

5. Verifikasi instalasi

```bash
pip list
```

## Business Dashboard

Aplikasi ini menyediakan dashboard analitik komprehensif yang memvisualisasikan data mahasiswa dan memberikan wawasan tentang faktor-faktor yang mempengaruhi tingkat dropout. Dashboard ini mencakup:

1. Distribusi status mahasiswa (Graduate, Dropout, Enrolled)
2. Analisis faktor-faktor kunci yang berkontribusi pada risiko dropout
3. Visualisasi statistik performa model
4. Perbandingan performa akademik antara mahasiswa dropout dan lulusan

Link untuk mengakses dashboard: [Student Dropout Risk Prediction](https://malikusfz-menyelesaikan-permasalahan-perusahaan-edut-app-yw31x7.streamlit.app/)

## Menjalankan Sistem Machine Learning

Sistem machine learning yang dikembangkan menggunakan algoritma XGBoost untuk memprediksi risiko dropout mahasiswa. Untuk menjalankan sistem ini secara lokal:

1. Pastikan virtual environment sudah aktif (jika belum, aktifkan dengan)

```bash
.\.env\Scripts\Activate.ps1
```

2. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan tersedia di browser pada alamat `http://localhost:8501`. Pengguna dapat memasukkan informasi mahasiswa di tab Prediction dan sistem akan menghasilkan prediksi risiko dropout beserta rekomendasi intervensi.

Link untuk mengakses prototype sistem: [Student Dropout Risk Prediction](https://malikusfz-menyelesaikan-permasalahan-perusahaan-edut-app-yw31x7.streamlit.app/)

## Conclusion

Berdasarkan analisis data dan model machine learning yang dikembangkan, dapat disimpulkan bahwa:

1. Model XGBoost yang dikembangkan mencapai akurasi 88.07% dalam memprediksi risiko dropout mahasiswa
2. Performa akademik semester pertama adalah prediktor terkuat untuk risiko dropout, diikuti oleh faktor keuangan
3. Mahasiswa dengan beasiswa menunjukkan tingkat kelulusan yang lebih tinggi dibandingkan mahasiswa tanpa beasiswa
4. Faktor-faktor seperti umur, status perkawinan, dan mode aplikasi juga memiliki pengaruh signifikan terhadap risiko dropout
5. Implementasi sistem peringatan dini dapat secara signifikan meningkatkan retensi mahasiswa dan mengurangi kerugian finansial institusi

### Rekomendasi Action Items

Berdasarkan hasil analisis, berikut adalah rekomendasi tindakan untuk mengurangi tingkat dropout mahasiswa:

- Implementasikan program mentoring khusus untuk mahasiswa yang menunjukkan performa rendah pada semester pertama
- Kembangkan program bantuan keuangan tambahan dan opsi pembayaran fleksibel untuk mahasiswa dengan kendala finansial
- Buat program intervensi akademik dini yang dipicu oleh hasil evaluasi pertama yang buruk
- Tingkatkan layanan dukungan untuk kelompok demografi dengan risiko dropout tinggi
- Selenggarakan sesi orientasi dan integrasi yang lebih komprehensif untuk mahasiswa baru
- Evaluasi dan revisi kurikulum mata kuliah dengan tingkat kegagalan tinggi
- Implementasikan pemantauan berkelanjutan menggunakan sistem prediksi risiko dropout yang telah dikembangkan
