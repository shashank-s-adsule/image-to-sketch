import numpy as np
import streamlit as st
import cv2
from PIL import Image

COLOR_FORMAT_OPTIONS = ["Grayscale","BGR","XYZ","YCrCb","YUV","HSV","HLS","Lab"]

_COLOR_FORMAT_MAP: dict = {
    "Grayscale": (cv2.COLOR_RGB2GRAY,  False),
    "BGR":       (cv2.COLOR_RGB2BGR,   True),
    "XYZ":       (cv2.COLOR_RGB2XYZ,   True),
    "YCrCb":     (cv2.COLOR_RGB2YCrCb, True),
    "YUV":       (cv2.COLOR_RGB2YUV,   True),
    "HSV":       (cv2.COLOR_RGB2HSV,   True),
    "HLS":       (cv2.COLOR_RGB2HLS,   True),
    "Lab":       (cv2.COLOR_RGB2Lab,   True),
}

def color_filters(image: np.ndarray, format: str = "Grayscale") -> np.ndarray:

    if format not in _COLOR_FORMAT_MAP:
        raise ValueError(
            f"Unknown color format '{format}'. Choose from {COLOR_FORMAT_OPTIONS}."
        )

    conversion_code, is_3ch = _COLOR_FORMAT_MAP[format]
    converted = cv2.cvtColor(image, conversion_code)

    if not is_3ch:
        # Single-channel (grayscale) -> broadcast to 3 channels for display
        converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2RGB)

    return converted

# Sketch pipeline 
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

def get_image_download_bytes(image: np.ndarray) -> bytes:
    encoded_success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not encoded_success:
        raise ValueError("Could not encode image for download.")
    return buffer.tobytes()

# Streamlit UI
def main() -> None:
    st.set_page_config(page_title="Sketchify", page_icon="🎨", layout="wide")

    st.markdown(
        "<style>div.block-container { padding-top: 1.75rem; }</style>",
        unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.markdown("""
        <div style="text-align: left; padding-bottom: 12px;
                    border-bottom: 1px solid #e0e0e0; margin-bottom: 16px;">
            <span style="font-size: 1.4rem; font-weight: 700; color: #4A90D9;">
                🎨 Sketchify
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Settings")

    output_mode = st.sidebar.radio(
        "Output Mode",
        options=["Sketch", "Color Filter"],
        index=0,
        help="Choose the type of output to generate.",
    )

    # Sketch-specific sliders
    if output_mode == "Sketch":
        k = st.sidebar.slider("Color quantization (K)", min_value=2, max_value=32, value=8, step=1)
        block_size = st.sidebar.slider("Edge block size", min_value=3, max_value=51, value=17, step=2)
        c = st.sidebar.slider("Adaptive threshold C", min_value=1, max_value=25, value=9, step=1)
        st.sidebar.caption(
            "Smaller edge block size → finer outlines. "
            "Larger → smoother edges."
        )
    else:
        k, block_size, c = 8, 17, 9  # unused defaults

    # Color-filter format selector
    if output_mode == "Color Filter":
        color_format = st.sidebar.selectbox(
            "Color Space",
            options=COLOR_FORMAT_OPTIONS,
            index=0,
            help="Select the OpenCV colour-space to apply to the image.",
        )
    else:
        color_format = "Grayscale"  # unused default

    # --- Main page ---
    st.title("Image to Sketch Converter")
    st.markdown(
        "Upload a photo and convert it into a sketch, apply a colour-space "
        "filter, or view a grayscale version."
    )

    uploaded_file = st.file_uploader(
        "Upload an image file",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_rgb = np.array(image.convert("RGB"))

        if output_mode == "Sketch":
            output = image_to_sketch(image, k=k, block_size=block_size, c=c)
            caption = "Sketch output"
            filename = "image_sketch.png"

        elif output_mode == "Color Filter":
            output = color_filters(img_rgb, format=color_format)
            caption = f"Color filter: {color_format}"
            filename = f"image_{color_format.lower()}.png"

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original image", width="stretch")
        with col2:
            st.image(output, caption=caption, width="stretch")

        download_bytes = get_image_download_bytes(output)
        st.download_button(
            label=f"Download as PNG",
            data=download_bytes,
            file_name=filename,
            mime="image/png",
        )
    else:
        st.info("Upload an image to get started.")

    # --- Sidebar footer: contact links ---
    st.sidebar.markdown(
        """
        <style>
        .contact-footer {
            position: fixed;
            bottom: 1rem;
            width: 230px;
            border-top: 1px solid #e0e0e0;
            padding-top: 10px;
        }
        .contact-footer a {
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            color: inherit;
            font-size: 0.85rem;
            margin-bottom: 6px;
            opacity: 0.75;
            transition: opacity 0.2s;
        }
        .contact-footer a:hover { opacity: 1; }
        .contact-footer svg { flex-shrink: 0; }
        </style>

        <div class="contact-footer">
            <a href="https://github.com/shashank-s-adsule" target="_blank">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                </svg>
                shashank-s-adsule
            </a>
            <a href="https://linkedin.com/in/shashank-adsule-a1b91a200" target="_blank">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="#0A66C2">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
                Shashank Adsule
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # How-it-works tooltip
    st.markdown("""
    <style>
    .info-icon {
        display: inline-block;
        width: 18px; height: 18px;
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
        top: 28px; left: 0;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        line-height: 1.5;
    }
    .info-icon:hover .tooltip-text { visibility: visible; opacity: 1; }
    </style>
    <div class="info-icon">i
        <div class="tooltip-text">
            <b>How it works</b><br><br>
            <b>Sketch:</b> Grayscale edge detection → K-means colour
            quantization → bilateral filter → combine with edge mask.<br><br>
            <b>Color Filter:</b> Converts the image to the selected OpenCV
            colour space (Grayscale, BGR, XYZ, YCrCb, YUV, HSV, HLS, Lab)
            and renders the raw channel values for visual inspection.<br><br>
            <b>Grayscale:</b> Simple single-channel luminance conversion.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()