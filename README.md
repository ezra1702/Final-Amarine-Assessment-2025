# Amarine Finall Test 
Nama : Christama Ezra Yudianto <br>
Nim : 245150307111009

## Milestone Riset Pendeteksian Objek Gambar Biota Laut

## 1. Perencanaan Awal

- Memahami dasar-dasar K-Means Clustering dan teknik pendeteksian objek.
- Mengumpulkan dataset gambar biota laut untuk eksperimen.
- Menentukan framework dan tools yang akan digunakan (OpenCV, scikit-learn, Matplotlib).

## 2. Analisis dan Desain

- Menentukan metode preprocessing gambar:
  - Konversi gambar ke RGB.
  - Normalisasi dan reshaping pixel data.
- Menggunakan **Elbow Method** untuk menentukan jumlah optimal cluster **K**.
- Merancang teknik clustering dengan K-Means untuk segmentasi warna.
- Mengembangkan metode visualisasi hasil clustering dengan Pie Chart.

## 3. Implementasi dan Pengujian

- Mengembangkan algoritma dalam Python menggunakan OpenCV dan K-Means Clustering.
- Menggunakan Elbow Method untuk menentukan nilai K terbaik.
- Menerapkan K-Means Clustering untuk segmentasi warna dalam gambar.
- Membuat visualisasi hasil clustering dalam bentuk gambar dan Pie Chart.
- Menguji algoritma pada dataset biota laut.
- Optimasi parameter K-Means untuk meningkatkan akurasi segmentasi.

## 4. Evaluasi dan Dokumentasi

- Membandingkan hasil dengan metode lain seperti DBSCAN atau Mean-Shift.
- Menulis laporan penelitian dan dokumentasi proyek di GitHub.
- Menyusun presentasi hasil riset.

## Tujuan Proyek

Proyek ini bertujuan untuk mengembangkan metode segmentasi dan analisis warna biota laut menggunakan **K-Means Clustering**. Dengan menerapkan teknik pemrosesan gambar dan pembelajaran mesin, proyek ini bertujuan untuk:
- Mengkategorikan warna dominan dalam gambar biota laut secara otomatis.
- Menganalisis distribusi warna untuk membantu identifikasi spesies biota laut.
- Mengeksplorasi efektivitas **Elbow Method** dalam menentukan jumlah optimal cluster (**K**) untuk segmentasi gambar.
- Menyediakan visualisasi hasil segmentasi menggunakan gambar yang telah diklasifikasikan dan **Pie Chart** warna dominan.
- **Menyediakan aplikasi berbasis web menggunakan Streamlit untuk mempermudah pengguna dalam melakukan analisis warna biota laut secara interaktif.**

## Diagram Alir Proses
<p align="center">
  <img src="images/diagram.png" alt="diagram" width="170">
</p>

## Implementasi Kode dalam Python 

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import platform
import psutil
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```
### Penjelasan Kode
Penjelasan Kode
Kode ini merupakan implementasi K-Means Clustering untuk segmentasi warna dalam gambar. Berikut adalah penjelasan dari setiap pustaka yang diimpor:

cv2 (OpenCV)

Digunakan untuk membaca dan memproses gambar, termasuk konversi warna dan manipulasi pixel.
numpy

Digunakan untuk manipulasi array dan operasi numerik, seperti mengubah gambar menjadi array pixel.
matplotlib.pyplot

Digunakan untuk menampilkan hasil visualisasi seperti gambar hasil segmentasi dan grafik Elbow Method.
time

Digunakan untuk mengukur waktu eksekusi program, yang berguna untuk optimasi performa algoritma.
platform

Digunakan untuk mendapatkan informasi tentang sistem operasi yang sedang digunakan.
psutil

Digunakan untuk memonitor penggunaan sumber daya sistem seperti CPU dan memori saat program berjalan.
sklearn.cluster.KMeans

Digunakan untuk mengimplementasikan algoritma K-Means Clustering dalam segmentasi warna.
sklearn.datasets.make_blobs

Digunakan untuk membuat dataset contoh dengan titik-titik data yang dikelompokkan ke dalam beberapa cluster (berguna untuk pengujian K-Means).

## SDLC Model: Agile

- **Sprint 1**: Studi literatur dan eksplorasi K-Means Clustering serta Elbow Method.
- **Sprint 2**: Implementasi algoritma awal, pengujian dataset kecil, dan debugging.
- **Sprint 3**: Optimasi algoritma, validasi dengan dataset lebih besar, dan analisis akurasi.
- **Sprint 4**: Dokumentasi, evaluasi, publikasi proyek di GitHub, serta penyusunan laporan riset.
- **Sprint 5**: Pengembangan aplikasi berbasis web dengan Streamlit dan deployment ke platform Streamlit Cloud.

## Tautan GitHub Proyek

[Masukkan tautan GitHub proyek di sini]

## Deployment Streamlit

Aplikasi ini dapat digunakan secara langsung melalui platform **Streamlit Cloud** untuk mempermudah analisis warna biota laut. [Masukkan tautan Streamlit di sini]

