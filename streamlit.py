import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image

def recolor_image_with_pie(uploaded_file):
    k = 4  

    image = Image.open(uploaded_file).convert("RGB") 
    image = np.array(image)
    
    pixels = image.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    recolored_pixels = kmeans.cluster_centers_[labels].reshape(image.shape).astype(int)
    
    unique, counts = np.unique(labels, return_counts=True)
    color_distribution = counts / counts.sum()
    
    dominant_colors = kmeans.cluster_centers_ / 255 

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))  
    
    axes[0].imshow(image)
    axes[0].set_title("Gambar Asli")
    axes[0].axis("off")
    
    axes[1].imshow(recolored_pixels)
    axes[1].set_title("Gambar dengan 4 Cluster Warna")
    axes[1].axis("off")

    axes[2].pie(color_distribution, labels=[f"Cluster {i+1}" for i in unique], colors=dominant_colors, autopct='%1.1f%%', startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'black'})
    axes[2].set_title("Distribusi Warna Dominan")
    
    st.pyplot(fig)

def run_streamlit():
    st.title("Pendeteksian Warna Biota Laut dengan K-Means")
    
    uploaded_file = st.file_uploader("Upload gambar biota laut", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar Asli", use_container_width=True)
        st.write("Sedang menganalisis warna...")
        recolor_image_with_pie(uploaded_file)
        st.write("Analisis selesai.")

if __name__ == "__main__":
    run_streamlit()
