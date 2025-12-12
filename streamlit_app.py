# app_rewritten.py
"""
Rewritten Streamlit app for AI Skin Lesion Classifier
- Robust model loading
- Safe Grad-CAM with fallbacks
- Conv layer selection dropdown built automatically
- OOD detection and safe overlay handling
- Clear error handling to avoid NameError/TypeError

Drop this file into your app folder and run with Streamlit.
"""

import os
import io
import zipfile
import gdown
from typing import Optional, List

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- CONFIG --------------------
DRIVE_FILE_ID = "1jeQ_juU_JjER89SCAoFseFRbVzZJNFYj"
MODEL_LOCAL_NAME = "best_resnet.keras"
MODEL_ZIP_NAME = "model_download.zip"
MODEL_IS_ZIP = False

CLASS_NAMES = ["Benign", "Malignant", "Nevus"]

CLASS_REPORTS = {
    "Benign": {
        "type": "Benign (non-cancerous)",
        "severity": "✅ Low Risk",
        "description": "Benign lesions are harmless...",
        "recommendation": "Monitor and routine check-up.",
        "urgency": "low",
        "color": "#28a745"
    },
    "Malignant": {
        "type": "Cancer (malignant)",
        "severity": "🚨 HIGH RISK - Cancer Detected",
        "description": "Malignant lesions require immediate attention...",
        "recommendation": "See a dermatologist immediately.",
        "urgency": "critical",
        "color": "#dc3545"
    },
    "Nevus": {
        "type": "Benign (typically)",
        "severity": "⚠️ Monitor Required",
        "description": "A nevus (mole) is usually benign...",
        "recommendation": "Monitor using ABCDE rules.",
        "urgency": "medium",
        "color": "#ffc107"
    }
}

st.set_page_config(page_title="🔬 AI Skin Lesion Classifier", page_icon="🔬", layout="wide")

# -------------------- UTILITIES --------------------

def download_model_from_drive(drive_id: str, dest: str, zip_dest: str = None, is_zip: bool = False) -> str:
    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    if is_zip:
        if zip_dest is None:
            raise ValueError("zip_dest must be provided when is_zip=True")
        gdown.download(url, zip_dest, quiet=False)
        return zip_dest
    else:
        gdown.download(url, dest, quiet=False)
        return dest


def safe_load_model(path: str):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error("Failed to load model. See logs for details.")
        st.exception(e)
        raise


def preprocess_image_pil(pil_img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    pil = pil_img.convert("RGB").resize(target_size)
    arr = np.array(pil).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def overlay_heatmap_on_image(original_image_np: np.ndarray, heatmap: np.ndarray, alpha=0.5):
    """Safe overlay - returns None if heatmap invalid."""
    if heatmap is None:
        return None
    heatmap = np.asarray(heatmap)
    if heatmap.size == 0:
        return None
    if np.isnan(heatmap).any() or np.isinf(heatmap).any():
        return None

    try:
        hmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
    except Exception:
        return None

    hmap = np.uint8(255 * (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8))
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    hmap_color = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)

    overlay = (original_image_np.astype("float32") * (1 - alpha) + hmap_color.astype("float32") * alpha)
    return np.clip(overlay, 0, 255).astype("uint8")


def get_all_conv_layer_names(model: tf.keras.Model) -> List[str]:
    names = []
    for layer in model.layers:
        lname = getattr(layer, "name", "")
        if "conv" in lname.lower() or "conv2d" in lname.lower():
            names.append(lname)
    return names


def get_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    names = get_all_conv_layer_names(model)
    return names[-1] if names else None


def check_if_out_of_distribution(predictions, confidence_threshold=70.0, entropy_threshold=1.1):
    preds = np.asarray(predictions).astype("float32")
    if preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]

    max_prob = float(np.max(preds)) * 100
    entropy = -np.sum(preds * np.log(preds + 1e-10))

    reasons = []
    if max_prob < confidence_threshold:
        reasons.append(f"Low confidence: {max_prob:.1f}% (<{confidence_threshold}%)")
    if entropy > entropy_threshold:
        reasons.append(f"High uncertainty: {entropy:.2f} (>{entropy_threshold})")

    return (len(reasons) > 0), reasons, entropy, max_prob


def simple_grad_cam(model: tf.keras.Model, image: np.ndarray, class_idx: int, layer_name: Optional[str]):
    """Robust simple Grad-CAM. Returns None if cannot compute."""
    if layer_name is None:
        return None

    try:
        layer = model.get_layer(layer_name)
    except Exception:
        return None

    try:
        grad_model = tf.keras.models.Model([model.inputs], [layer.output, model.output])
    except Exception:
        return None

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        conv_outputs, predictions = grad_model(image_tensor)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[0, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()
    return heatmap_np

# -------------------- MODEL LOADING --------------------
@st.cache_resource(show_spinner=False)
def get_model(download_if_missing: bool = True):
    if os.path.exists(MODEL_LOCAL_NAME):
        model_path = MODEL_LOCAL_NAME
    else:
        if not download_if_missing:
            raise FileNotFoundError(MODEL_LOCAL_NAME)
        st.info("Downloading model...")
        if MODEL_IS_ZIP:
            downloaded = download_model_from_drive(DRIVE_FILE_ID, MODEL_LOCAL_NAME, zip_dest=MODEL_ZIP_NAME, is_zip=True)
            with zipfile.ZipFile(downloaded, "r") as z:
                z.extractall(".")
            if os.path.exists(MODEL_LOCAL_NAME):
                model_path = MODEL_LOCAL_NAME
            else:
                candidates = [f for f in os.listdir(".") if f.endswith('.keras') or f.endswith('.h5')]
                model_path = candidates[0] if candidates else None
                if model_path is None:
                    raise FileNotFoundError("Model not found after extracting zip.")
        else:
            download_model_from_drive(DRIVE_FILE_ID, MODEL_LOCAL_NAME, is_zip=False)
            model_path = MODEL_LOCAL_NAME

    model = safe_load_model(model_path)
    return model

# -------------------- UI --------------------
st.title("🔬 AI Skin Lesion Classifier")
st.markdown("**3-Class Classification: Benign • Malignant • Nevus**")
st.markdown("---")

with st.spinner("Loading model..."):
    try:
        model = get_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error("❌ Model loading error. See logs for details.")
        st.stop()

# Sidebar - robust conv layer detection and selection
st.sidebar.header("⚙️ Settings")
conv_layer_names = get_all_conv_layer_names(model)
if not conv_layer_names:
    conv_layer_names = ["(no_conv_layers)"]

detected_last_conv = get_last_conv_layer(model) or conv_layer_names[0]

selected_layer = st.sidebar.selectbox("Choose conv layer for Grad-CAM", conv_layer_names, index=conv_layer_names.index(detected_last_conv) if detected_last_conv in conv_layer_names else 0)

layer_name = selected_layer if selected_layer != "(no_conv_layers)" else None
show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)

# OOD Detection Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Image Validation")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 50, 90, 70)
entropy_threshold = st.sidebar.slider("Uncertainty Threshold", 0.5, 1.5, 1.1, step=0.1)

# Main upload
st.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader("Choose a clear, well-lit image of the skin lesion", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("📤 Please upload a skin lesion image to begin analysis")
    st.markdown("---")
    st.markdown("### 🎯 What This System Can Detect:")
    for class_name in CLASS_NAMES:
        with st.expander(f"📋 {class_name}"):
            info = CLASS_REPORTS[class_name]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(f"**Severity:** {info['severity']}")
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Recommendation:** {info['recommendation']}")
    st.markdown("---")
    st.stop()

# If file uploaded
try:
    image_data = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
except Exception as e:
    st.error("❌ Could not read the uploaded image.")
    st.exception(e)
    st.stop()

col1, col2 = st.columns([1, 1.5])
with col1:
    st.markdown("#### 📸 Uploaded Image")
    st.image(pil_img, use_column_width=True)

with col2:
    st.markdown("#### 📊 Analysis Results")
    with st.spinner("🔍 Analyzing image..."):
        input_img = preprocess_image_pil(pil_img, target_size=(224, 224))
        try:
            preds = model.predict(input_img, verbose=0)
        except Exception as e:
            st.error("Model prediction failed.")
            st.exception(e)
            st.stop()

        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        pred_idx = int(np.argmax(preds[0]))
        pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
        pred_prob = float(preds[0, pred_idx]) * 100

        is_ood, ood_reasons, entropy, max_prob = check_if_out_of_distribution(preds[0], confidence_threshold=confidence_threshold, entropy_threshold=entropy_threshold)

    if is_ood:
        st.error("⚠️ **Invalid Image Detected**")
        st.warning("\n".join(ood_reasons))
        with st.expander("🔍 Technical Details"):
            st.write(f"Top prediction: {pred_class}")
            st.write(f"Confidence: {pred_prob:.2f}%")
            st.write(f"Entropy: {entropy:.2f}")
            prob_data = [{"Class": name, "Probability": f"{float(preds[0,i])*100:.2f}%"} for i,name in enumerate(CLASS_NAMES)]
            st.dataframe(pd.DataFrame(prob_data), hide_index=True)
        st.stop()

    st.success(f"✅ Valid lesion detected")
    class_info = CLASS_REPORTS.get(pred_class, CLASS_REPORTS[CLASS_NAMES[0]])
    st.markdown(f"<h3 style='color: {class_info['color']};'>{class_info['severity']}</h3>", unsafe_allow_html=True)
    st.markdown(f"### {pred_class}")
    st.markdown(f"**Confidence:** {pred_prob:.1f}%")
    st.progress(pred_prob / 100.0)
    st.markdown(f"**Type:** {class_info['type']}")

# Full-width info
st.markdown("---")
col3, col4 = st.columns(2)
with col3:
    st.markdown("### 📋 About This Condition")
    st.info(class_info['description'])
with col4:
    st.markdown("### 💊 Recommended Action")
    if class_info['urgency'] == 'critical':
        st.error(class_info['recommendation'])
    elif class_info['urgency'] == 'medium':
        st.warning(class_info['recommendation'])
    else:
        st.success(class_info['recommendation'])

# Probabilities
st.markdown("---")
st.markdown("### 📊 Detailed Probability Breakdown")
prob_data = [{"Condition": name, "Probability": f"{float(preds[0,i])*100:.2f}%"} for i,name in enumerate(CLASS_NAMES)]
prob_df = pd.DataFrame(prob_data)
prob_df = prob_df.sort_values("Probability", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
st.dataframe(prob_df, use_container_width=True, hide_index=True)

# Grad-CAM
if show_gradcam:
    st.markdown("---")
    st.markdown("### 🔥 AI Focus Map (Grad-CAM)")
    st.markdown("*Shows which areas the AI analyzed to make its prediction*")

    if layer_name is None:
        st.warning("No convolutional layer available for Grad-CAM on this model.")
    else:
        try:
            with st.spinner("Generating heatmap..."):
                heatmap = simple_grad_cam(model, input_img.astype('float32'), pred_idx, layer_name)

                if heatmap is None or np.max(heatmap) == 0:
                    st.warning("⚠️ Could not generate heatmap (no activation detected or unsupported layer).")
                else:
                    orig_np = np.array(pil_img.convert('RGB'))
                    overlay = overlay_heatmap_on_image(orig_np, heatmap, alpha=0.6)

                    col5, col6, col7 = st.columns([1,1,1])
                    with col5:
                        st.image(pil_img, caption='Original', use_column_width=True)
                    with col6:
                        fig, ax = plt.subplots()
                        ax.imshow(heatmap, cmap='jet')
                        ax.axis('off')
                        st.pyplot(fig)
                        st.caption('Heatmap')
                    with col7:
                        if overlay is None:
                            st.warning('Could not create overlay image.')
                        else:
                            st.image(overlay, caption='Overlay', use_column_width=True)
        except Exception as e:
            st.error('❌ Grad-CAM generation failed.')
            st.exception(e)

# Downloadable report
st.markdown('---')
report_text = f"""SKIN LESION ANALYSIS REPORT
{'='*70}

PREDICTION RESULTS:
Detected Condition: {pred_class}
Confidence: {pred_prob:.2f}%
Type: {class_info['type']}
Severity: {class_info['severity']}

VALIDATION METRICS:
Entropy: {entropy:.2f} (threshold: {entropy_threshold})
Max Probability: {max_prob:.2f}%
Status: Valid skin lesion image

DETAILED PROBABILITIES:
"""
for i,name in enumerate(CLASS_NAMES):
    report_text += f"{name}: {float(preds[0,i])*100:.2f}%\n"

report_text += f"""
\nDESCRIPTION:\n{class_info['description']}\n\nRECOMMENDATION:\n{class_info['recommendation']}\n
{'='*70}\nIMPORTANT DISCLAIMER:\nThis is an AI screening tool and should NOT replace professional medical diagnosis. Always consult a qualified dermatologist.\n\nReport generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"""

st.download_button(label='📥 Download Full Report (TXT)', data=report_text, file_name=f"skin_lesion_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt", mime='text/plain')

st.markdown('---')
st.warning('⚠️ **MEDICAL DISCLAIMER:** This AI tool is for screening purposes only. Always consult a qualified dermatologist for proper diagnosis and treatment.')

