# app.py
import os
import io
import zipfile
import tempfile
import gdown
from typing import Optional

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd

DRIVE_FILE_ID = "1jeQ_juU_JjER89SCAoFseFRbVzZJNFYj"
MODEL_LOCAL_NAME = "best_resnet.keras"                 
MODEL_ZIP_NAME = "model_download.zip"               
MODEL_IS_ZIP = False 

# ===== 3 CLASSES FROM YOUR NEW MODEL =====
CLASS_NAMES = [
    "Benign",
    "Malignant", 
    "Nevus"
]

CLASS_REPORTS = {
    "Benign": {
        "type": "Benign (non-cancerous)",
        "severity": "✅ Low Risk",
        "description": (
            "Benign lesions include seborrheic keratoses, dermatofibromas, and vascular lesions. "
            "These are harmless, non-cancerous skin growths that are very common, especially with aging. "
            "They appear as brown, black, or tan growths with various textures."
        ),
        "recommendation": (
            "✅ **No treatment necessary in most cases.** Monitor for any unusual changes in appearance. "
            "Removal is optional for cosmetic reasons or if irritated by clothing. "
            "Schedule a routine check-up with your dermatologist."
        ),
        "urgency": "low",
        "color": "#28a745"
    },
    
    "Malignant": {
        "type": "Cancer (malignant)",
        "severity": "🚨 HIGH RISK - Cancer Detected",
        "description": (
            "Malignant lesions include melanoma, basal cell carcinoma, and actinic keratosis. "
            "These are cancerous or pre-cancerous conditions that require immediate medical attention. "
            "Melanoma is the most dangerous form and can spread rapidly if not treated early."
        ),
        "recommendation": (
            "🚨 **URGENT: See a dermatologist immediately.** Early detection and treatment are critical. "
            "Treatment may include surgical excision, Mohs surgery, immunotherapy, targeted therapy, "
            "or radiation depending on the type and stage. Do not delay - schedule an appointment today."
        ),
        "urgency": "critical",
        "color": "#dc3545"
    },
    
    "Nevus": {
        "type": "Benign (typically)",
        "severity": "⚠️ Monitor Required",
        "description": (
            "A nevus (mole) is a benign growth of melanocytes. Most people have 10-40 moles. "
            "While usually harmless, moles should be monitored for changes that could indicate melanoma. "
            "Common moles are typically uniform in color and symmetrical."
        ),
        "recommendation": (
            "👁️ **Monitor using the ABCDE rule:**\n\n"
            "- **A**symmetry: One half doesn't match the other\n"
            "- **B**order: Irregular, scalloped, or poorly defined\n"
            "- **C**olor: Varied colors (brown, black, tan, red, white, blue)\n"
            "- **D**iameter: Larger than 6mm (pencil eraser)\n"
            "- **E**volving: Changes in size, shape, or color\n\n"
            "Schedule a dermatologist check-up if you notice any changes. "
            "Photograph the mole and monitor it monthly."
        ),
        "urgency": "medium",
        "color": "#ffc107"
    }
}

st.set_page_config(
    page_title="🔬 AI Skin Lesion Classifier", 
    page_icon="🔬",
    layout="wide"
)

# -------------------------
# Utility Functions
# -------------------------
def download_model_from_drive(drive_id: str, dest: str, zip_dest: str = None, is_zip: bool = False):
    """Download model file from Google Drive (direct download)."""
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
    """Load a Keras model with a friendly error message."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error("Failed to load model. See logs for details.")
        st.write("Model loading error:", e)
        raise

def preprocess_image_pil(pil_img: Image.Image, target_size=(224,224)):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def overlay_heatmap_on_image(original_image_np, heatmap, alpha=0.5):
    """Overlay heatmap with robust error handling."""
    # ✅ FIX: Validate heatmap
    if heatmap is None or heatmap.size == 0:
        return original_image_np
    
    # ✅ FIX: Ensure heatmap is 2D
    if len(heatmap.shape) > 2:
        heatmap = np.squeeze(heatmap)
    
    if len(heatmap.shape) != 2:
        print(f"Invalid heatmap dimensions: {heatmap.shape}")
        return original_image_np
    
    # ✅ FIX: Ensure float32 for cv2.resize
    heatmap = heatmap.astype(np.float32)
    
    try:
        hmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
    except Exception as e:
        print(f"Resize error: {e}, heatmap shape: {heatmap.shape}")
        return original_image_np
    
    hmap = np.uint8(255 * hmap)
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    hmap_color = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = (original_image_np.astype("float32") * (1 - alpha) + hmap_color.astype("float32") * alpha)
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    
    return overlay

def get_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    """Find the last convolutional layer name in the model."""
    last_conv = None
    for layer in model.layers:
        if hasattr(layer, "output_shape"):
            shp = layer.output_shape
            if isinstance(shp, tuple) and len(shp) == 4:
                name = layer.name.lower()
                if "conv" in name or "conv2d" in name:
                    last_conv = layer.name
    return last_conv

def check_if_out_of_distribution(predictions, confidence_threshold=70.0, entropy_threshold=1.1):
    """
    Detect if an image is out-of-distribution (not a skin lesion)
    
    Args:
        predictions: Model predictions (softmax outputs)
        confidence_threshold: Minimum confidence % for top prediction
        entropy_threshold: Maximum entropy (lower for 3 classes)
    
    Returns:
        (is_ood, reasons, entropy, max_prob)
    """
    max_prob = np.max(predictions) * 100
    
    reasons = []
    
    # Check confidence
    if max_prob < confidence_threshold:
        reasons.append(f"Low confidence: {max_prob:.1f}% (need ≥{confidence_threshold}%)")
    
    # Check entropy (for 3 classes, max entropy is ~1.1)
    entropy = -np.sum(predictions * np.log(predictions + 1e-10))
    
    if entropy > entropy_threshold:
        reasons.append(f"High uncertainty: {entropy:.2f} (threshold: {entropy_threshold})")
    
    is_ood = len(reasons) > 0
    
    return is_ood, reasons, entropy, max_prob

def simple_grad_cam(model, image, class_idx, layer_name='conv5_block3_out'):
    """Simple Grad-CAM implementation with robust error handling."""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
    except Exception as e:
        print(f"Error creating grad model: {e}")
        return None
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        
        # Handle multiple outputs
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        
        loss = predictions[0, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        print("No gradients computed")
        return None
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight channels
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ✅ FIX: Ensure heatmap is 2D
    if len(heatmap.shape) > 2:
        heatmap = tf.reduce_mean(heatmap, axis=-1)  # Average extra dimensions
    
    # Convert to numpy
    heatmap = heatmap.numpy()
    
    # ✅ FIX: Check heatmap validity
    if heatmap.size == 0 or heatmap.ndim != 2:
        print(f"Invalid heatmap shape: {heatmap.shape}")
        return None
    
    # Normalize
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        print("Heatmap is all zeros")
        return None
    
    return heatmap

@st.cache_resource(show_spinner=False)
def get_model(download_if_missing: bool = True):
    if os.path.exists(MODEL_LOCAL_NAME):
        model_path = MODEL_LOCAL_NAME
    else:
        if not download_if_missing:
            raise FileNotFoundError(f"{MODEL_LOCAL_NAME} not found.")
        st.info("Downloading model...")
        if MODEL_IS_ZIP:
            downloaded = download_model_from_drive(DRIVE_FILE_ID, MODEL_LOCAL_NAME, zip_dest=MODEL_ZIP_NAME, is_zip=True)
            with zipfile.ZipFile(downloaded, "r") as z:
                z.extractall(".")
            if os.path.exists(MODEL_LOCAL_NAME):
                model_path = MODEL_LOCAL_NAME
            else:
                candidates = [f for f in os.listdir(".") if f.endswith(".keras") or os.path.isdir(f)]
                if candidates:
                    model_path = candidates[0]
                else:
                    raise FileNotFoundError("Could not locate model inside the extracted zip.")
        else:
            downloaded = download_model_from_drive(DRIVE_FILE_ID, MODEL_LOCAL_NAME, is_zip=False)
            model_path = MODEL_LOCAL_NAME

    model = safe_load_model(model_path)
    return model

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔬 AI Skin Lesion Classifier")
st.markdown("**3-Class Classification: Benign • Malignant • Nevus**")
st.markdown("---")

with st.spinner("Loading model..."):
    try:
        model = get_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error("❌ Model loading error.")
        st.stop()

# Sidebar
st.sidebar.header("⚙️ Settings")
default_layer = "conv5_block3_out"
detected_last_conv = get_last_conv_layer(model)
if detected_last_conv is None:
    detected_last_conv = default_layer

layer_name = st.sidebar.text_input(
    "Conv layer for Grad-CAM", 
    value=detected_last_conv,
    help="Leave as default for automatic detection"
)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)

# OOD Detection Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Image Validation")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)", 
    min_value=50, 
    max_value=90, 
    value=70,
    help="Minimum confidence to accept prediction"
)
entropy_threshold = st.sidebar.slider(
    "Uncertainty Threshold", 
    min_value=0.5, 
    max_value=1.5, 
    value=1.1,
    step=0.1,
    help="Maximum uncertainty (entropy) for 3 classes"
)

# Main upload section
st.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a clear, well-lit image of the skin lesion",
    type=["jpg", "png", "jpeg"],
    help="Supported formats: JPG, PNG, JPEG"
)

if uploaded_file is not None:
    try:
        image_data = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        st.error("❌ Could not read the uploaded image.")
        st.stop()

    # Layout
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("#### 📸 Uploaded Image")
        st.image(pil_img, use_column_width=True)

    with col2:
        st.markdown("#### 📊 Analysis Results")
        
        with st.spinner("🔍 Analyzing image..."):
            input_img = preprocess_image_pil(pil_img, target_size=(224,224))
            preds = model.predict(input_img, verbose=0)
            
            if preds.ndim == 1:
                preds = np.expand_dims(preds, axis=0)
            
            pred_idx = int(np.argmax(preds[0]))
            pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
            pred_prob = float(preds[0, pred_idx]) * 100
            
            # OOD Detection
            is_ood, ood_reasons, entropy, max_prob = check_if_out_of_distribution(
                preds[0], 
                confidence_threshold=confidence_threshold,
                entropy_threshold=entropy_threshold
            )
        
        if is_ood:
            st.error("⚠️ **Invalid Image Detected**")
            
            st.warning(
                "**This doesn't appear to be a valid skin lesion image.**\n\n"
                "**Detected issues:**\n" + 
                "\n".join(f"- {r}" for r in ood_reasons) + 
                "\n\n**Please upload:**\n"
                "✅ A close-up photo of a skin lesion\n"
                "✅ Clear, well-lit image\n"
                "✅ Dermatoscopic or clinical photograph\n"
                "✅ Image in focus\n\n"
                "**Do NOT upload:**\n"
                "❌ Random objects\n"
                "❌ Full body photos\n"
                "❌ Blurry or dark images"
            )
            
            with st.expander("🔍 Technical Details"):
                st.write(f"**Top prediction:** {pred_class}")
                st.write(f"**Confidence:** {pred_prob:.2f}%")
                st.write(f"**Entropy:** {entropy:.2f}")
                
                prob_data = []
                for i, name in enumerate(CLASS_NAMES):
                    prob = float(preds[0, i]) * 100
                    prob_data.append({"Class": name, "Probability": f"{prob:.2f}%"})
                st.dataframe(pd.DataFrame(prob_data), hide_index=True)
            
            st.stop()
        
        # Valid image
        st.success(f"✅ Valid lesion detected")

        class_info = CLASS_REPORTS[pred_class]
        
        # Display prediction
        st.markdown(
            f"<h3 style='color: {class_info['color']};'>{class_info['severity']}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"### {pred_class}")
        
        # Confidence meter
        st.markdown(f"**Confidence:** {pred_prob:.1f}%")
        st.progress(pred_prob / 100)
        
        st.markdown(f"**Type:** {class_info['type']}")

    # Full-width sections
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 📋 About This Condition")
        st.info(class_info['description'])
    
    with col4:
        st.markdown("### 💊 Recommended Action")
        urgency = class_info['urgency']
        if urgency == "critical":
            st.error(class_info['recommendation'])
        elif urgency == "medium":
            st.warning(class_info['recommendation'])
        else:
            st.success(class_info['recommendation'])

    # Probabilities
    st.markdown("---")
    st.markdown("### 📊 Detailed Probability Breakdown")
    
    prob_data = []
    for i, name in enumerate(CLASS_NAMES):
        prob = float(preds[0, i]) * 100
        prob_data.append({"Condition": name, "Probability": f"{prob:.2f}%"})
    
    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values("Probability", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
    
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Grad-CAM
    # Grad-CAM
    if show_gradcam:
        st.markdown("---")
        st.markdown("### 🔥 AI Focus Map (Grad-CAM)")
        st.markdown("*Shows which areas the AI analyzed to make its prediction*")
        
        # Try multiple layers if first fails
        layers_to_try = [layer_name, 'conv5_block3_out', 'conv5_block2_out', 'conv4_block6_out']
        heatmap = None
        used_layer = None
        
        for try_layer in layers_to_try:
            try:
                _ = model.get_layer(try_layer)
                with st.spinner(f"Generating heatmap using {try_layer}..."):
                    heatmap = simple_grad_cam(model, input_img.astype("float32"), pred_idx, try_layer)
                    
                    if heatmap is not None and np.max(heatmap) > 0:
                        used_layer = try_layer
                        break
            except Exception as e:
                continue
        
        if heatmap is not None and used_layer is not None:
            try:
                orig_np = np.array(pil_img.convert("RGB"))
                overlay = overlay_heatmap_on_image(orig_np, heatmap, alpha=0.6)
                
                if used_layer != layer_name:
                    st.info(f"ℹ️ Using layer: {used_layer}")
                
                col5, col6, col7 = st.columns([1, 1, 1])
                
                with col5:
                    st.image(pil_img, caption="Original", use_column_width=True)
                
                with col6:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(heatmap, cmap='jet')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("Heatmap")
                
                with col7:
                    st.image(overlay, caption="Overlay", use_column_width=True)
                
            except Exception as e:
                st.error("❌ Grad-CAM visualization failed.")
                with st.expander("Show error"):
                    st.code(str(e))
        else:
            st.warning("⚠️ Could not generate Grad-CAM heatmap. This may happen with certain layer configurations.")
            except Exception as e:
                st.error("❌ Grad-CAM generation failed.")
                with st.expander("Show error"):
                    st.code(str(e))

    # Download report
    st.markdown("---")
    st.markdown("### 📥 Download Report")
    
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
    for i, name in enumerate(CLASS_NAMES):
        prob = float(preds[0, i]) * 100
        report_text += f"{name}: {prob:.2f}%\n"

    report_text += f"""
DESCRIPTION:
{class_info['description']}

RECOMMENDATION:
{class_info['recommendation']}

{'='*70}
IMPORTANT DISCLAIMER:
This is an AI screening tool and should NOT replace professional 
medical diagnosis. Always consult a qualified dermatologist.

Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Accuracy: 89.52% (on test dataset)
Image File: {uploaded_file.name}
"""

    st.download_button(
        label="📥 Download Full Report (TXT)",
        data=report_text,
        file_name=f"skin_lesion_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # Disclaimer
    st.markdown("---")
    st.warning(
        "⚠️ **MEDICAL DISCLAIMER:** This AI tool is for screening purposes only. "
        "Always consult a qualified dermatologist for proper diagnosis and treatment."
    )

else:
    # No image uploaded
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
    st.markdown("### 📈 Model Performance")
    st.success("✅ **89.52% accuracy** on test set")
    st.info("📊 Trained on 21,000 balanced images")
    st.info("🏗️ ResNet50 + SE Attention architecture")
    
    st.markdown("---")
    st.markdown("### 📸 Image Requirements")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**✅ DO Upload:**")
        st.markdown("""
        - Close-up photos of skin lesions
        - Clear, well-lit images
        - Dermatoscopic images
        - Clinical photographs
        - Images in focus
        """)
    
    with col_b:
        st.markdown("**❌ DON'T Upload:**")
        st.markdown("""
        - Random objects
        - Full body photos
        - Blurry or dark images
        - Screenshots
        - Non-skin images
        """)
