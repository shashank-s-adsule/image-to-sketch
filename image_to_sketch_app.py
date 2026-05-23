import io

import numpy as np
import streamlit as st
import cv2
from PIL import Image


def color_quantization(img: np.ndarray, k: int) -> np.ndarray:
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    return result.reshape(img.shape)


def image_to_sketch(image: Image.Image, k: int, block_size: int, c: int) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )

    quantized = color_quantization(img, max(2, k))
    filtered = cv2.bilateralFilter(quantized, 9, 250, 250)
    sketch = cv2.bitwise_and(filtered, filtered, mask=edge_mask)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

    return sketch


def color_filters(image: np.ndarray, format="grayscale") -> np.ndarray:
    # add open tabe where we we can select for for respective color format 
    '''cv2.COLOR_BGR2RGB
cv2.COLOR_RGB2GRAY
cv2.COLOR_RGB2BGR
cv2.COLOR_RGB2XYZ
cv2.COLOR_RGB2YCrCb
cv2.COLOR_RGB2YUV
cv2.COLOR_RGB2HSV
cv2.COLOR_RGB2HLS
cv2.COLOR_RGB2Lab'''

    return None

def get_image_download_bytes(image: np.ndarray) -> bytes:
    encoded_success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not encoded_success:
        raise ValueError("Could not encode image for download.")
    return buffer.tobytes()


def main() -> None:
    st.set_page_config(page_title="Image to Sketch", page_icon="🖼️", layout="wide")

    st.sidebar.markdown(
        """
        <div style="text-align: left; padding-bottom: 12px; border-bottom: 1px solid #e0e0e0; margin-bottom: 16px;">
            <span style="font-size: 1.4rem; font-weight: 700; color: #4A90D9;">🎨 Sketchify</span><br/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Sketch Settings")

    # Output mode selector
    output_mode = st.sidebar.radio(
        "Output Mode",
        options=["Sketch", "Grayscale"],
        index=0,
        help="Choose how the processed image should look.",
    )

    # Sketch-specific sliders — only shown when Sketch mode is active
    if output_mode == "Sketch":
        k = st.sidebar.slider("Color quantization (K)", min_value=2, max_value=32, value=8, step=1)
        block_size = st.sidebar.slider("Edge block size", min_value=3, max_value=51, value=17, step=2)
        c = st.sidebar.slider("Adaptive threshold C", min_value=1, max_value=25, value=9, step=1)
        st.sidebar.write(
            "Use a smaller edge block size for more detailed outlines and a larger value for smoother edges."
        )
    else:
        k, block_size, c = 8, 17, 9  # defaults, unused in grayscale mode

    st.title("Image to Sketch Converter")
    st.markdown(
        "Upload a photo and convert it into a sketch-style image using OpenCV color quantization, bilateral filtering, and edge masking."
    )

    uploaded_file = st.file_uploader(
        "Upload an image file",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        #########
        if output_mode == "Grayscale":
            img_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            caption = "Grayscale output"
            filename = "image_grayscale.png"
        else:
            output = image_to_sketch(image, k=k, block_size=block_size, c=c)
            caption = "Sketch output"
            filename = "image_sketch.png"

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original image", use_column_width=True)
        with col2:
            st.image(output, caption=caption, use_column_width=True)

        download_bytes = get_image_download_bytes(output)
        st.download_button(
            label=f"Download {output_mode.lower()} as PNG",
            data=download_bytes,
            file_name=filename,
            mime="image/png",
        )
    else:
        st.info("Upload an image to get started.")

    st.markdown("""
    <style>
    .info-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background-color: #444;
        color: white;
        text-align: center;
        font-size: 12px;
        line-height: 18px;
        cursor: pointer;
        position: relative;
        font-weight: bold;
    }

    .info-icon .tooltip-text {
        visibility: hidden;
        width: 420px;
        background-color: #222;
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1;
        top: 28px;
        left: 0;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        line-height: 1.5;
    }

    .info-icon:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    </style>

    <div class="info-icon">
        i
        <div class="tooltip-text">
            <b>How it works</b><br><br>
            1. Convert the uploaded image to grayscale and extract edges with adaptive thresholding.<br>
            2. Apply K-means color quantization to reduce the image palette.<br>
            3. Smooth the color output with a bilateral filter.<br>
            4. Combine the edge mask with the filtered image to create the sketch-style result.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
