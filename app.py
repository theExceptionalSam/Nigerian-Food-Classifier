import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import plotly.graph_objects as go

# -------------------------------------------------------
# PAGE CONFIGURATION — must be the very first Streamlit call
# -------------------------------------------------------
st.set_page_config(
    page_title="Nigerian Food Classifier",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# LOAD MODEL AND CLASS INDICES
# Cached so it only loads once, not on every interaction
# -------------------------------------------------------

@st.cache_resource
def load_model():
    import tensorflow as tf
    import gdown

    model_path = 'nigerian_food_final.keras'

    if not os.path.exists(model_path):
        with st.spinner("Downloading model... (first load only)"):
            FILE_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE'
            url     = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)
    return model

# -------------------------------------------------------
# PREPROCESSING FUNCTION
# Must match exactly what was used during training
# -------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepares a PIL image for EfficientNetB0 inference.

    Args:
        image : PIL Image object (any size, any mode)

    Returns:
        numpy array of shape (1, 224, 224, 3), normalised
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # Resize to model's expected input size
    image = image.convert('RGB')
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Apply EfficientNet-specific normalisation
    img_array = preprocess_input(img_array)

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------

def predict(model, img_array: np.ndarray, idx_to_class: dict, top_k: int = 3):
    """
    Runs inference and returns top-k predictions.

    Args:
        model        : loaded Keras model
        img_array    : preprocessed image array
        idx_to_class : index to class name mapping
        top_k        : number of top predictions to return

    Returns:
        list of (class_name, confidence_percent) tuples
    """
    predictions   = model.predict(img_array, verbose=0)[0]
    top_indices   = np.argsort(predictions)[::-1][:top_k]

    results = [
        (idx_to_class[idx], float(predictions[idx]) * 100)
        for idx in top_indices
    ]
    return results

# -------------------------------------------------------
# UI STYLING
# -------------------------------------------------------

def apply_custom_styles():
    st.markdown("""
    <style>
        /* Main container */
        .main { padding-top: 1rem; }

        /* Header */
        .app-header {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
        }

        /* Prediction card */
        .prediction-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin: 0.5rem 0;
            border-left: 5px solid #1f77b4;
        }

        .prediction-rank {
            font-size: 0.8rem;
            color: #888;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .prediction-name {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1a1a2e;
            margin: 0.2rem 0;
        }

        .prediction-confidence {
            font-size: 1.1rem;
            font-weight: 600;
        }

        /* Upload area */
        .upload-section {
            text-align: center;
            padding: 1rem 0;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #aaa;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# CONFIDENCE BAR CHART
# -------------------------------------------------------

def render_confidence_chart(predictions: list):
    """
    Renders a horizontal bar chart of top-k predictions using Plotly.
    """
    names        = [p[0] for p in reversed(predictions)]
    confidences  = [p[1] for p in reversed(predictions)]
    colours      = ['#1f77b4', '#aec7e8', '#d4e6f1'][:len(predictions)]
    colours      = list(reversed(colours))

    fig = go.Figure(go.Bar(
        x=confidences,
        y=names,
        orientation='h',
        marker=dict(color=colours),
        text=[f"{c:.1f}%" for c in confidences],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text="Confidence Scores", font=dict(size=14)),
        xaxis=dict(
            title="Confidence (%)",
            range=[0, min(max(confidences) + 15, 105)],
            showgrid=True,
            gridcolor='#eee'
        ),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=200,
        margin=dict(l=10, r=60, t=40, b=40),
        font=dict(family="sans-serif", size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------

def main():
    apply_custom_styles()

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🍲 Nigerian Food Classifier</h1>
        <p style="color: #666; font-size: 1rem;">
            Upload a photo of Nigerian food and the model will identify it.<br>
            Trained on 18 classes using EfficientNetB0 — 85.75% accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Load assets
    with st.spinner("Loading model..."):
        model        = load_model()
        idx_to_class = load_class_indices()

    # Supported classes info
    with st.expander("See all 18 supported food classes"):
        classes = list(idx_to_class.values())
        cols    = st.columns(3)
        for i, cls in enumerate(sorted(classes)):
            cols[i % 3].markdown(f"• {cls}")

    st.markdown("### Upload Your Image")

    # File uploader
    uploaded_file = st.file_uploader(
        label="Choose a food image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )

    if uploaded_file is not None:

        # Load and display image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("**Uploaded Image**")
            st.image(image, use_column_width=True)
            st.caption(
                f"Size: {image.size[0]} x {image.size[1]}px  |  "
                f"Format: {image.format or uploaded_file.type.split('/')[1].upper()}"
            )

        with col2:
            st.markdown("**Prediction Results**")

            with st.spinner("Analysing image..."):
                img_array   = preprocess_image(image)
                predictions = predict(model, img_array, idx_to_class, top_k=3)

            # Top prediction — primary result
            top_name, top_conf = predictions[0]
            confidence_colour  = (
                "#27ae60" if top_conf >= 80 else
                "#f39c12" if top_conf >= 50 else
                "#e74c3c"
            )

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-rank">Top Prediction</div>
                <div class="prediction-name">{top_name}</div>
                <div class="prediction-confidence" style="color: {confidence_colour};">
                    {top_conf:.1f}% confident
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence indicator
            if top_conf >= 80:
                st.success("High confidence prediction")
            elif top_conf >= 50:
                st.warning("Moderate confidence — consider the alternatives below")
            else:
                st.error("Low confidence — image may be ambiguous or outside training classes")

        # Full confidence chart below both columns
        st.markdown("### Top 3 Predictions")
        render_confidence_chart(predictions)

        # Detailed breakdown table
        st.markdown("### Detailed Breakdown")
        for rank, (name, conf) in enumerate(predictions, 1):
            col_rank, col_name, col_bar, col_conf = st.columns([0.5, 2.5, 4, 1])
            col_rank.markdown(f"**#{rank}**")
            col_name.markdown(f"{name}")
            col_bar.progress(int(conf))
            col_conf.markdown(f"**{conf:.1f}%**")

    else:
        # Empty state
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem 1rem;
            background: #f8f9fa;
            border-radius: 12px;
            border: 2px dashed #ddd;
            margin: 1rem 0;
        ">
            <div style="font-size: 3rem;">📸</div>
            <div style="font-size: 1.1rem; color: #666; margin-top: 0.5rem;">
                Upload a food image to get started
            </div>
            <div style="font-size: 0.85rem; color: #aaa; margin-top: 0.3rem;">
                JPG, PNG or WEBP accepted
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        Nigerian Food Classifier · EfficientNetB0 · 18 Classes · 85.75% Accuracy
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
