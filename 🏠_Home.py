import streamlit as st
from src.pipeline import advanced_cartoon_pipeline
from PIL import Image
import numpy as np
import io

st.set_page_config(
    page_title="NPR Graph Studio: Advanced Abstraction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Elegant Workspace Design
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    h1 { color: #1e3a8a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
    h2, h3 { color: #334155; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; color: #64748b; }
    .stTabs [aria-selected="true"] { color: #1e3a8a; border-bottom-color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

st.title("Non-Photorealistic Rendering & Color Space Quantization Studio")
st.markdown("### Structural Scale-Space Separation Engine using Machine Learning & Iterative Filter Pyramids")

main_tab, theory_tab = st.tabs(["🔬 Live Execution Pipeline", "📝 Architectural Core & Mathematics"])

with main_tab:
    st.markdown("---")
    
    # --- Sidebar Hyper-parameters ---
    st.sidebar.header("Algorithmic Configuration")
    st.sidebar.subheader("Scale-Space Segmentation")
    num_bilateral = st.sidebar.slider("Bilateral Filter Iterations", 2, 12, 6, step=1)
    k_colors = st.sidebar.slider("K-Means Quantization Clusters (K)", 2, 16, 8, step=1)
    
    st.sidebar.subheader("Local Topology Boundary Control")
    edge_block = st.sidebar.slider("Adaptive Block Size", 3, 21, 9, step=2)
    edge_c = st.sidebar.slider("Luminance Tuning Constant (C)", 1, 20, 7, step=1)
    
    # --- Input Source Selector ---
    st.subheader("Data Ingestion Subsystem")
    input_mode = st.radio(
        "Select Ingestion Source Protocol:", 
        ["Static File Ingestion", "Webcam Snapshot Interface"]
    )
    
    image_bytes = None
    
    if input_mode == "Static File Ingestion":
        uploaded_file = st.file_uploader("Ingest Research Matrix Target Asset (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image_bytes = uploaded_file.read()
            
    else:
        st.markdown("#### Hardware Sensor Interface")
        # Native browser snapshot component
        cam_shot = st.camera_input("Position your subject and click 'Take Photo'")
        if cam_shot:
            image_bytes = cam_shot.read()

    # --- Processing & Synthesis Execution ---
    if image_bytes:
        with st.spinner("Processing advanced matrix arrays via K-Means and Scaled Pyramids..."):
            orig, gray, edge_mask, color_filtered, final = advanced_cartoon_pipeline(
                image_bytes, num_bilateral, k_colors, edge_block, edge_c
            )
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Source Array Space")
            st.image(orig, use_container_width=True)
        with col2:
            st.subheader("Synthesized Mathematical Reconstruction")
            st.image(final, use_container_width=True)
            
            # Direct In-Memory Stream compilation for browser file downloads
            result_pil = Image.fromarray(final)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                label="💾 Export Synthesized Asset Matrix",
                data=buf.getvalue(),
                file_name="npr_synthesis_output.png",
                mime="image/png"
            )
            
        st.markdown("### Intermediate Scale-Space Pipeline Review")
        t1, t2, t3 = st.tabs(["1. Luminance Analysis", "2. Gaussian Adaptive Boundaries", "3. K-Means Quantized Surface"])
        with t1:
            st.image(gray, caption="Monochromatic Luminance Matrix Array", use_container_width=True)
        with t2:
            st.image(edge_mask, caption="Anti-Aliased Boundary Vector Matrix", use_container_width=True)
        with t3:
            st.image(color_filtered, caption="Quantized Topological Color Field", use_container_width=True)

with theory_tab:
    st.markdown("## Academic Framework & Pipeline Mechanics")
    st.markdown("""
    This project explores Non-Photorealistic Rendering (NPR) by treating an image as a combination of structural boundaries and flat color regions. 

    ### 1. Scale-Space Smoothing via Iterative Bilateral Filtering
    Standard Gaussian filtering blurs high-frequency detail uniformly across an image, destroying edge structures. The Bilateral Filter limits blurring by factoring in both spatial proximity and radiometric color similarity:

    $$I^{\\text{filtered}}(x) = \\frac{1}{W_p} \\sum_{x_i \\in \\Omega} I(x_i) g_s(\\|x_i - x\\|) f_r(\\|I(x_i) - I(x)\\|)$$

    Where $g_s$ represents spatial Gaussian weight distribution and $f_r$ represents radiometric range similarity weight distribution. To maximize performance, this engine runs an **Iterative Pyramidal Loop**: downsampling the image matrix array to filter macrostructures efficiently, running multiple small-kernel bilateral passes to stabilize gradient regions, and then upsampling back to scale.

    ### 2. Unsupervised Machine Learning Color Quantization
    Instead of continuous gradients, cartoon painting relies on discrete color paletting. We treat the RGB image pixels as a 3-dimensional data cluster and apply **K-Means Clustering** optimization to find $K$ ideal center color vectors by minimizing inertia:

    $$\\arg\\min_{\\mathbf{S}} \\sum_{i=1}^{K} \\sum_{\\mathbf{x} \\in S_i} \\|\\mathbf{x} - \\boldsymbol{\\mu}_i\\|^2$$

    Every single coordinate voxel pixel is replaced with its nearest cluster centroid vector $\\boldsymbol{\\mu}_i$, generating a highly realistic painted appearance.

    ### 3. High-Contrast Adaptive Boundary Tracking
    Structural lines are determined using a dynamic window Gaussian Thresholding profile:

    $$T(x,y) = G_{\\sigma}(x,y) * L(x,y) - C$$

    By evaluating localized neighborhood standard deviations rather than a fixed global value, the pipeline prevents line dropouts caused by uneven shadows or complex lighting variations.
    """)