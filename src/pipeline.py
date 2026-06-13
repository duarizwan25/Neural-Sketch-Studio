import cv2
import numpy as np
from sklearn.cluster import KMeans

def quantize_colors(img_rgb, k=8):
    pixels = img_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_
    quantized = centers[labels].reshape(img_rgb.shape)
    return quantized

def advanced_cartoon_pipeline(image_bytes, num_bilateral=7, k_colors=8, edge_block_size=9, edge_c=9):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    img_color = img_rgb.copy()
    img_down = cv2.pyrDown(img_color)
    
    for _ in range(num_bilateral):
        img_down = cv2.bilateralFilter(img_down, d=5, sigmaColor=9, sigmaSpace=7)
        
    img_blur = cv2.pyrUp(img_down)
    if img_blur.shape != img_rgb.shape:
        img_blur = cv2.resize(img_blur, (img_rgb.shape[1], img_rgb.shape[0]))
        
    img_quantized = quantize_colors(img_blur, k=k_colors)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_smoothed = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_smoothed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, edge_block_size, edge_c
    )
    
    edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
    edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)
    
    final_cartoon = (img_quantized.astype(float) * edge_mask).astype(np.uint8)
    
    return img_rgb, gray, edges, img_quantized, final_cartoon
