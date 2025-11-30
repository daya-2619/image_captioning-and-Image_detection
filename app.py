import streamlit as st
from PIL import Image
from vision_module import analyze_image

st.title("🖼️ Image Captioning + Object Detection (BLIP + DETR)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        caption, detections, annotated = analyze_image(uploaded)

    st.subheader("📝 Caption")
    st.write(caption)

    st.subheader("🎯 Detections")
    for d in detections:
        st.write(f"- **{d['label']}** ({d['score']:.2f})")

    st.subheader("📌 Annotated Image")
    st.image(annotated, use_column_width=True)
