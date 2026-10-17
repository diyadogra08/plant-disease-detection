import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a clear photo of a plant leaf to identify "
    "potential diseases using Artificial Intelligence."
)

MODEL_PATH = Path(__file__).parent / "model" / "plant_saved_model"

@st.cache_resource
def load_model():
    return tf.saved_model.load(str(MODEL_PATH))

try:
    model = load_model()
    st.success("AI model loaded successfully!")
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    if img_array.max() > 1:
        img_array = img_array / 255.0

    if st.button("Detect Plant Disease"):
        try:
            infer = model.signatures["serving_default"]
            input_name = list(
                infer.structured_input_signature[1].keys()
            )[0]

            output = infer(**{
                input_name: tf.convert_to_tensor(img_array)
            })

            predictions = list(output.values())[0].numpy()
            predicted_class = int(np.argmax(predictions[0]))

            st.success("🌿 Analysis Complete!")
            st.subheader("Prediction Result")
            class_names = json.load(open(
            Path(__file__).parent / "class_names.json"
            ))
            st.write(f"Predicted disease: {class_names[predicted_class]}")
            confidence = float(np.max(predictions[0])) * 100
            confidence = float(np.max(predictions[0])) * 100

            st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
            )
          

        except Exception as e:
            st.error(f"Prediction failed: {e}")


            st.divider()

st.subheader("About This Project")
st.write("""
This project uses Deep Learning and the MobileNetV2
architecture to classify plant leaf images into
plant disease categories.

The model was trained on a labelled dataset of
plant leaf images.
""")

st.caption("Developed as an AI/ML student project.")