# Amarine Finall Test 
Nama : Christama Ezra Yudianto  
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
```plaintext
![ElbowImage](images/elbow.png)
```


## Implementasi Kode dalam Python 

### 1. Mengimpor Pustaka yang Dibutuhkan
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import platform
import psutil
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```

**Penjelasan:**
- `cv2`: Untuk membaca dan memproses gambar.
- `numpy`: Untuk manipulasi array numerik.
- `matplotlib.pyplot`: Untuk visualisasi data.
- `time`: Untuk mengukur performa algoritma.
- `platform`: Untuk mendapatkan informasi sistem operasi.
- `psutil`: Untuk memonitor penggunaan CPU dan memori.
- `sklearn.cluster.KMeans`: Untuk algoritma K-Means Clustering.
- `sklearn.datasets.make_blobs`: Untuk membuat dataset dummy.

### 2. Konversi Gambar ke Format RGB dan Ekstraksi Piksel
```python
# Membaca gambar dan mengubah ke format RGB
image_path = "images/aethaloperca_rogaa_11.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Ubah gambar menjadi kumpulan piksel (flatten)
pixels = image.reshape(-1, 3)
height, width, channels = image.shape

print(f"Bentuk Awal (3D) → ({height}, {width}, {channels})")

print(f"{height} = tinggi gambar")
print(f"{width} = lebar gambar")
print(f"{channels} = jumlah kanal warna (RGB)")
print(f"Bentuk Akhir (2D) → {pixels.shape}")

```
**Output:**
```plaintext
Bentuk Awal (3D) → (432, 650, 3)
432 = tinggi gambar
650 = lebar gambar
3 = jumlah kanal warna (RGB)
Bentuk Akhir (2D) → (280800, 3)
```
**Penjelasan:**
1. Bentuk Awal (3D) → (432, 650, 3)
 - Gambar awal memiliki tinggi = 432 piksel
 - Gambar memiliki lebar = 650 piksel
 - Gambar memiliki 3 kanal warna (RGB), di mana setiap piksel 
 - memiliki tiga nilai untuk warna Merah (R), Hijau (G), dan Biru= (B)
2.  Bentuk Akhir (2D) → (280800, 3)
 - Setelah gambar diratakan (flattened), bentuknya berubah dari (432, 650, 3) menjadi (280800, 3).
 - 280800 berasal dari 432 × 650, yaitu jumlah total piksel dalam gambar.
 - Setiap piksel tetap memiliki 3 nilai warna (RGB) sehingga bentuk akhirnya menjadi (jumlah piksel, 3).

### 3. Menentukan Cluster Terbaik Dengan Elbow Method
```python
# Hitung inertia untuk berbagai nilai K
max_k = 10  # Batas atas jumlah cluster
inertia = []
k_values = range(1, max_k + 1)
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    inertia.append(kmeans.inertia_)
    print(f"K = {k}, Inertia = {kmeans.inertia_:.2f}")
```
**Output:**
```plaintext
K = 1, Inertia = 866037899.82
K = 2, Inertia = 427127533.45
K = 3, Inertia = 283660527.88
K = 4, Inertia = 201961651.82
K = 5, Inertia = 140273876.40
K = 6, Inertia = 116479413.24
K = 7, Inertia = 93609222.28
K = 8, Inertia = 81971134.45
K = 9, Inertia = 73143205.75
K = 10, Inertia = 66512469.12
```

**Penjelasan:**
- K = jumlah klaster dalam K-Means.
- Inertia = total jarak kuadrat antara titik data dan pusat klaster.
- Semakin kecil inertia, semakin baik pemisahan klaster.
- Saat K meningkat, inertia menurun, tetapi dengan laju yang semakin lambat.
- Gunakan metode "Elbow" untuk menentukan K optimal, yaitu titik di mana penurunan inertia mulai melambat.
- Dari data, K = 4 atau K = 5 mungkin optimal.

### 4. Grafik Elbow Method untuk Menentukan K Optimal
```python
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method untuk Menentukan K Optimal')
plt.xticks(k_values)
plt.grid()
plt.show()
```
**Output:**
```plaintext

```
**Penjelasan:**
- Menggunakan **inertia** untuk menentukan nilai **K** optimal.
- Memplot nilai **K** terhadap nilai **distorsi** untuk menemukan **titik siku**.

### 5. Menggunakan K-Means untuk Segmentasi Gambar
```python
def apply_kmeans(pixels, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

segmented_image = apply_kmeans(pixels, k=4)
plt.imshow(segmented_image.astype(int))
plt.axis("off")
plt.show()
```
**Penjelasan:**
- Menerapkan **K-Means Clustering** pada data pixel gambar.
- Menggunakan jumlah cluster **K = 4**.
- Mengonversi kembali hasil segmentasi ke bentuk gambar.

### 6. Visualisasi Warna Dominan dalam Bentuk Pie Chart
```python
def plot_color_pie(pixels, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_ / 255
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    plt.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%")
    plt.title("Distribusi Warna Dominan")
    plt.show()

plot_color_pie(pixels, k=4)
```
**Penjelasan:**
- Menghitung warna dominan dari hasil clustering.
- Memplot distribusi warna menggunakan **Pie Chart**.

## SDLC Model: Agile

- **Sprint 1**: Studi literatur dan eksplorasi K-Means Clustering serta Elbow Method.
- **Sprint 2**: Implementasi algoritma awal, pengujian dataset kecil, dan debugging.
- **Sprint 3**: Optimasi algoritma, validasi dengan dataset lebih besar, dan analisis akurasi.
- **Sprint 4**: Dokumentasi, evaluasi, publikasi proyek di GitHub, serta penyusunan laporan riset.
- **Sprint 5**: Pengembangan aplikasi berbasis web dengan Streamlit dan deployment ke platform Streamlit Cloud.

## Tautan GitHub Proyek
[Masukkan tautan GitHub proyek di sini]

## Deployment Streamlit
Aplikasi ini dapat digunakan secara langsung melalui platform **Streamlit Cloud**. [Masukkan tautan Streamlit di sini]
