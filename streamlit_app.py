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

DRIVE_FILE_ID = "191J7lKgDMBQ-urrk4LVH3afaE89ciiRT"  
MODEL_LOCAL_NAME = "my_model.keras"                 
MODEL_ZIP_NAME = "model_download.zip"               
MODEL_IS_ZIP = False 

# ===== 6 ACTUAL CLASSES FROM YOUR MODEL =====
CLASS_NAMES = [
    "Actinic Keratosis / Intraepithelial Carcinoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Melanocytic Nevus",
    "Melanoma",
    "Vascular / Dermatofibroma"
]

CLASS_REPORTS = {
    "Actinic Keratosis / Intraepithelial Carcinoma": {
        "type": "Pre-cancerous",
        "severity": " Moderate Risk",
        "description": (
            "Actinic keratosis is a precancerous skin growth caused by sun damage. "
            "While not cancer itself, it can develop into squamous cell carcinoma if left untreated. "
            "These appear as rough, scaly patches on sun-exposed areas."
        ),
        "recommendation": (
            " **Medical evaluation recommended.** Treatment options include cryotherapy, "
            "topical medications, or photodynamic therapy. Regular monitoring is essential."
        ),
        "urgency": "medium",
        "color": "#ffc107"
    },
    
    "Basal Cell Carcinoma": {
        "type": "Cancer (malignant)",
        "severity": " High Risk - Cancer Detected",
        "description": (
            "Basal cell carcinoma is the most common form of skin cancer. It grows slowly "
            "and rarely spreads to other parts of the body, but can cause local tissue damage "
            "if left untreated. Often appears as a pearly or waxy bump."
        ),
        "recommendation": (
            " **Immediate dermatologist consultation required.** Early treatment has excellent "
            "success rates. Treatment typically involves surgical removal, Mohs surgery, or radiation."
        ),
        "urgency": "high",
        "color": "#dc3545"
    },
    
    "Benign Keratosis": {
        "type": "Benign (non-cancerous)",
        "severity": " Low Risk",
        "description": (
            "Benign keratoses (including seborrheic keratoses) are harmless, non-cancerous "
            "skin growths. They're very common, especially with aging, and appear as brown, "
            "black, or tan growths with a waxy, scaly texture."
        ),
        "recommendation": (
            " **No treatment necessary in most cases.** Monitor for changes in appearance. "
            "Removal is optional for cosmetic reasons or if irritated by clothing."
        ),
        "urgency": "low",
        "color": "#28a745"
    },
    
    "Melanocytic Nevus": {
        "type": "Benign (typically)",
        "severity": " Low Risk",
        "description": (
            "A melanocytic nevus (common mole) is a benign growth of melanocytes. "
            "Most people have 10-40 moles on their body. They're usually harmless but "
            "should be monitored for changes that could indicate melanoma."
        ),
        "recommendation": (
            "  **Monitor using the ABCDE rule:**\n\n"
            "- **A**symmetry: One half doesn't match the other\n"
            "- **B**order: Irregular, scalloped, or poorly defined\n"
            "- **C**olor: Varied colors (brown, black, tan, red, white, blue)\n"
            "- **D**iameter: Larger than 6mm (pencil eraser)\n"
            "- **E**volving: Changes in size, shape, or color\n\n"
            "Schedule a check-up if you notice any changes."
        ),
        "urgency": "low",
        "color": "#28a745"
    },
    
    "Melanoma": {
        "type": "Cancer (malignant)",
        "severity": " HIGH RISK - Serious Cancer Detected",
        "description": (
            "Melanoma is the most dangerous form of skin cancer. It develops in melanocytes "
            "(pigment-producing cells) and can spread rapidly to other organs if not caught early. "
            "Early detection and treatment are critical for survival."
        ),
        "recommendation": (
            " **URGENT: See a dermatologist immediately.** Melanoma requires prompt medical "
            "attention. Treatment may include surgical excision, immunotherapy, targeted therapy, "
            "or radiation. Early-stage melanoma has a high cure rate with proper treatment."
        ),
        "urgency": "critical",
        "color": "#8b0000"
    },
    
    "Vascular / Dermatofibroma": {
        "type": "Benign (non-cancerous)",
        "severity": " Low Risk",
        "description": (
            "This category includes vascular lesions (such as hemangiomas or cherry angiomas) "
            "and dermatofibromas. These are benign growths that are harmless. Vascular lesions "
            "appear as red or purple spots, while dermatofibromas are firm bumps in the skin."
        ),
        "recommendation": (
            " **No treatment needed unless causing discomfort.** These lesions are benign. "
            "Removal is optional for cosmetic reasons. Monitor for unusual changes."
        ),
        "urgency": "low",
        "color": "#28a745"
    }
}

st.set_page_config(
    page_title=" AI Skin Cancer Early Detection", 
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
    hmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
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

def check_if_out_of_distribution(predictions, confidence_threshold=70.0, entropy_threshold=1.5):
    """
    Detect if an image is out-of-distribution (not a skin lesion)
    
    Args:
        predictions: Model predictions (softmax outputs)
        confidence_threshold: Minimum confidence % for top prediction
        entropy_threshold: Maximum entropy (higher = more uncertain)
    
    Returns:
        (is_ood, reasons)
    """
    max_prob = np.max(predictions) * 100
    
    reasons = []
    
    # Check confidence
    if max_prob < confidence_threshold:
        reasons.append(f"Low confidence: {max_prob:.1f}% (need ≥{confidence_threshold}%)")
    
    # Check entropy (spread of probabilities)
    entropy = -np.sum(predictions * np.log(predictions + 1e-10))
    
    if entropy > entropy_threshold:
        reasons.append(f"High uncertainty: {entropy:.2f} (threshold: {entropy_threshold})")
    
    is_ood = len(reasons) > 0
    
    return is_ood, reasons, entropy, max_prob

def grad_cam_plus_plus(model, image, class_idx, layer_name):
    """Robust Grad-CAM++ implementation."""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)

        if isinstance(predictions, (list, tuple)):
            preds_candidate = None
            for p in predictions:
                try:
                    p_tensor = tf.convert_to_tensor(p)
                    if p_tensor.shape.rank is not None and p_tensor.shape.rank >= 1:
                        preds_candidate = p_tensor
                        break
                except Exception:
                    continue
            if preds_candidate is None:
                preds_candidate = tf.convert_to_tensor(predictions[0])
            predictions = preds_candidate
        else:
            predictions = tf.convert_to_tensor(predictions)
        loss = predictions[0, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    grads_sq = tf.square(grads)
    grads_cu = grads_sq * grads

    alpha_num = grads_sq
    alpha_denom = 2 * grads_sq + tf.reduce_sum(conv_outputs * grads_cu, axis=(1,2), keepdims=True)
    alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))

    alphas = alpha_num / alpha_denom
    weights = tf.reduce_sum(alphas * tf.maximum(grads, 0), axis=(1,2))
    weights = tf.reshape(weights, [-1,1,1,tf.shape(conv_outputs)[-1]])

    grad_cam_output = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.maximum(grad_cam_output, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = tf.squeeze(heatmap)
    return heatmap.numpy()

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
st.title("AI Skin Cancer Early Detection System")
st.markdown("---")

with st.spinner("Loading ..."):
    try:
        model = get_model()
        st.success(" Model loaded")
    except Exception as e:
        st.error(" error.")
        st.stop()

# Sidebar
st.sidebar.header(" Settings")
default_layer = "conv5_block3_3_conv"
detected_last_conv = get_last_conv_layer(model)
if detected_last_conv is None:
    detected_last_conv = default_layer

layer_name = st.sidebar.text_input(
    "Conv layer for Grad-CAM", 
    value=detected_last_conv,
    help="Leave as default for automatic detection"
)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM++ Heatmap", value=True)

# OOD Detection Settings
st.sidebar.markdown("---")
st.sidebar.markdown("###  Image Validation")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)", 
    min_value=50, 
    max_value=90, 
    value=70,
    help="Minimum confidence to accept prediction"
)
entropy_threshold = st.sidebar.slider(
    "Uncertainty Threshold", 
    min_value=1.0, 
    max_value=2.5, 
    value=1.5,
    step=0.1,
    help="Maximum uncertainty (entropy) allowed"
)
# Main upload section
st.markdown("###  Upload Image")
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
        st.error(" Could not read the uploaded image.")
        st.stop()

    # Layout: Image on left, results on right
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("####  Uploaded Image")
        st.image(pil_img, use_column_width=True)

    with col2:
        st.markdown("####  Analysis Results")
        
        with st.spinner(" Analyzing image..."):
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
            st.error(" **Invalid Image Detected**")
            
            st.warning(
                "**This doesn't appear to be a valid skin lesion image.**\n\n"
                "**Detected issues:**\n" + 
                "\n".join(f"- {r}" for r in ood_reasons) + 
                "\n\n**Please upload:**\n"
                " A close-up photo of a skin lesion\n"
                " Clear, well-lit image\n"
                " Dermatoscopic or clinical photograph\n"
                " Image in focus\n\n"
                "**Do NOT upload:**\n"
                " Random objects (balls, food, animals, etc.)\n"
                " Full body photos\n"
                " Blurry or dark images\n"
                " Screenshots or edited images"
            )
            
            with st.expander("🔍 Technical Details (for debugging)"):
                st.write(f"**Top prediction:** {pred_class}")
                st.write(f"**Confidence:** {pred_prob:.2f}%")
                st.write(f"**Entropy:** {entropy:.2f}")
                st.write(f"**Confidence threshold:** {confidence_threshold}%")
                st.write(f"**Entropy threshold:** {entropy_threshold}")
                
                st.markdown("**All probabilities:**")
                prob_data = []
                for i, name in enumerate(CLASS_NAMES):
                    prob = float(preds[0, i]) * 100
                    prob_data.append({"Class": name, "Probability": f"{prob:.2f}%"})
                st.dataframe(pd.DataFrame(prob_data), hide_index=True)
            
            st.info(" **Tip:** Adjust the validation thresholds in the sidebar if you believe this is a valid image.")
            st.stop()
        
        # Valid image - continue normally
        st.success(f" Valid lesion detected")

        # Get class info
        class_info = CLASS_REPORTS[pred_class]
        
        # Display prediction with color coding
        st.markdown(
            f"<h3 style='color: {class_info['color']};'>{class_info['severity']}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"### {pred_class}")
        
        # Confidence meter
        st.markdown(f"**Confidence:** {pred_prob:.1f}%")
        st.progress(pred_prob / 100)
        
        st.markdown(f"**Type:** {class_info['type']}")

    # Full-width sections below
    st.markdown("---")
    
    # Description and Recommendation
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("###  About This Condition")
        st.info(class_info['description'])
    
    with col4:
        st.markdown("###  Recommended Action")
        urgency = class_info['urgency']
        if urgency in ["high", "critical"]:
            st.error(class_info['recommendation'])
        elif urgency == "medium":
            st.warning(class_info['recommendation'])
        else:
            st.success(class_info['recommendation'])

    # Detailed probabilities
    st.markdown("---")
    st.markdown("###  Detailed Probability Breakdown")
    
    prob_data = []
    for i, name in enumerate(CLASS_NAMES):
        prob = float(preds[0, i]) * 100 if i < preds.shape[1] else 0.0
        prob_data.append({"Condition": name, "Probability": f"{prob:.2f}%"})
    
    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values("Probability", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
    
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Grad-CAM visualization
    if show_gradcam:
        st.markdown("---")
        st.markdown("###  AI Focus Map (Grad-CAM++)")
        st.markdown("*This heatmap shows which areas the AI focused on to make its prediction*")
        
        try:
            _ = model.get_layer(layer_name)
        except Exception:
            auto = get_last_conv_layer(model)
            if auto:
                st.warning(f"Layer '{layer_name}' not found. Using '{auto}' instead.")
                layer_name = auto
            else:
                st.error("Could not find a convolutional layer for Grad-CAM.")
                layer_name = None

        if layer_name:
            try:
                with st.spinner("Generating heatmap..."):
                    heatmap = grad_cam_plus_plus(model, input_img.astype("float32"), pred_idx, layer_name)
                    orig_np = np.array(pil_img.convert("RGB"))
                    overlay = overlay_heatmap_on_image(orig_np, heatmap, alpha=0.5)
                
                col5, col6, col7 = st.columns([1, 1, 1])
                with col5:
                    st.image(pil_img, caption="Original", use_column_width=True)
                with col6:
                    fig, ax = plt.subplots()
                    ax.imshow(heatmap, cmap='jet')
                    ax.axis('off')
                    st.pyplot(fig)
                    st.caption("Heatmap")
                with col7:
                    st.image(overlay, caption="Overlay", use_column_width=True)
                
            except Exception as e:
                st.error(" Grad-CAM generation failed.")
                with st.expander("Show error details"):
                    st.code(str(e))

    # Download report
    st.markdown("---")
    st.markdown("###  Download Report")
    
    report_text = f"""SKIN LESION ANALYSIS REPORT
{'='*70}

PREDICTION RESULTS:
Detected Condition: {pred_class}
Confidence: {pred_prob:.2f}%
Type: {class_info['type']}
Severity: {class_info['severity']}

VALIDATION METRICS:
Entropy: {entropy:.2f} (threshold: {entropy_threshold})
Confidence Threshold: {confidence_threshold}%
Status: Valid skin lesion image

DETAILED PROBABILITIES:
"""
    for i, name in enumerate(CLASS_NAMES):
        prob = float(preds[0, i]) * 100 if i < preds.shape[1] else 0.0
        report_text += f"{name}: {prob:.2f}%\n"

    report_text += f"""
DESCRIPTION:
{class_info['description']}

RECOMMENDATION:
{class_info['recommendation']}

{'='*70}
IMPORTANT DISCLAIMER:
This is an AI screening tool and should NOT replace professional 
medical diagnosis. Always consult a qualified dermatologist for 
proper evaluation and treatment recommendations.

Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Accuracy: 96.94% (on test dataset)
Image File: {uploaded_file.name}
"""

    st.download_button(
        label=" Download Full Report (TXT)",
        data=report_text,
        file_name=f"skin_lesion_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # Disclaimer
    st.markdown("---")
    st.warning(
        " **MEDICAL DISCLAIMER:** This AI tool is designed for screening purposes only "
        "and should not be used as a substitute for professional medical advice, diagnosis, "
        "or treatment. Always seek the advice of a qualified dermatologist or healthcare "
        "provider with any questions regarding a skin condition."
    )

else:
    # No image uploaded - show info
    st.info(" Please upload a skin lesion image to begin analysis")
    
    st.markdown("---")
    st.markdown("###  What This System Can Detect:")
    
    for class_name in CLASS_NAMES:
        with st.expander(f" {class_name}"):
            info = CLASS_REPORTS[class_name]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(f"**Severity:** {info['severity']}")
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Recommendation:** {info['recommendation']}")
    
    st.markdown("---")
    st.markdown("###  Model Performance")
    st.success(" **96.94% accuracy** ")
    st.info(" Trained on dermatoscopic images")
    st.info(" Based on ResNet50 architecture with custom attention mechanism")
    
    st.markdown("---")
    st.markdown("###  Image Requirements")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("** DO Upload:**")
        st.markdown("""
        - Close-up photos of skin lesions
        - Clear, well-lit images
        - Dermatoscopic images
        - Clinical photographs
        - Images in focus
        """)
    
    with col_b:
        st.markdown("** DON'T Upload:**")
        st.markdown("""
        - Random objects (balls, food, etc.)
        - Full body photos
        - Blurry or dark images
        - Screenshots or edited images
        - Non-skin images
        """)
