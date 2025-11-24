import streamlit as st
from PIL import Image
import requests
import io

FASTAPI_URL = "http://api:8000/colorize"

st.set_page_config(page_title="Image Colorizer", layout="centered")
st.title("Image Colorizer")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    if st.button("Colorize"):
        with st.spinner("Processing..."):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            files = {"file": ("image.png", img_bytes, "image/png")}
            response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                colorized_img = Image.open(io.BytesIO(response.content))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original image", use_container_width=True)
                with col2:
                    st.image(colorized_img, caption="Colorized image", use_container_width=True)
            else:
                st.error(f"Server error: {response.status_code}")
    else:
        st.image(image, caption="Original image", use_container_width=True)
else:
    st.info("Upload a grayscale image to begin.")