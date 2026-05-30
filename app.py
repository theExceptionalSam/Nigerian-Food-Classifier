import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nigerian Food Classifier",
    page_icon="🍲",
    layout="centered"
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

@st.cache_resource
def load_model():
    import gdown
    import onnxruntime as ort

    model_path = 'nigerian_food_model.onnx'

    if not os.path.exists(model_path):
        with st.spinner("Downloading model... (first load only, please wait)"):
            FILE_ID = '1FU6UfaAckJ534tTqwSnyUMDXlr48A62j'
            url     = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, model_path, quiet=False)

    session = ort.InferenceSession(model_path)
    return session


@st.cache_data
def load_class_indices():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}


# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    image     = image.convert('RGB')
    image     = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)

    # EfficientNet preprocessing without TensorFlow
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return img_array


# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

def predict(session, img_array: np.ndarray, idx_to_class: dict, top_k: int = 3):
    input_name  = session.get_inputs()[0].name
    outputs     = session.run(None, {input_name: img_array})
    predictions = outputs[0][0]

    top_indices = np.argsort(predictions)[::-1][:top_k]
    results     = [
        (idx_to_class[idx], float(predictions[idx]) * 100)
        for idx in top_indices
    ]
    return results


# -------------------------------------------------------
# CONFIDENCE CHART
# -------------------------------------------------------

def render_chart(predictions: list):
    names       = [p[0] for p in reversed(predictions)]
    scores      = [p[1] for p in reversed(predictions)]
    colours     = ['#1f77b4', '#aec7e8', '#d4e6f1']
    colours     = list(reversed(colours[:len(predictions)]))

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation='h',
        marker=dict(color=colours),
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Confidence Scores",
        xaxis=dict(
            title="Confidence (%)",
            range=[0, min(max(scores) + 15, 110)],
            showgrid=True,
            gridcolor='#eee'
        ),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=200,
        margin=dict(l=10, r=60, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------
# MAIN APP
# -------------------------------------------------------

def main():

    st.markdown("""
        <h1 style='text-align:center'>🍲 Nigerian Food Classifier</h1>
        <p style='text-align:center; color:#666;'>
            Upload a food photo — the model identifies it from 18 Nigerian food classes.<br>
            Built with EfficientNetB0 · 85.75% accuracy · 18 classes
        </p>
    """, unsafe_allow_html=True)

    st.divider()

    # Load model and class map
    with st.spinner("Loading model..."):
        session      = load_model()
        idx_to_class = load_class_indices()

    # Show supported classes
    with st.expander("See all 18 supported food classes"):
        classes = sorted(idx_to_class.values())
        cols    = st.columns(3)
        for i, cls in enumerate(classes):
            cols[i % 3].markdown(f"• {cls}")

    st.markdown("### Upload Your Image")

    uploaded_file = st.file_uploader(
        label="Choose a food image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Supported: JPG, JPEG, PNG, WEBP"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("**Uploaded Image**")
            st.image(image, use_column_width=True)
            st.caption(f"Size: {image.size[0]} x {image.size[1]}px")

        with col2:
            st.markdown("**Prediction Results**")

            with st.spinner("Analysing image..."):
                img_array   = preprocess_image(image)
                predictions = predict(session, img_array, idx_to_class, top_k=3)

            top_name, top_conf = predictions[0]

            if top_conf >= 80:
                colour = "#27ae60"
                label  = "High confidence"
            elif top_conf >= 50:
                colour = "#f39c12"
                label  = "Moderate confidence"
            else:
                colour = "#e74c3c"
                label  = "Low confidence"

            st.markdown(f"""
            <div style="
                background:#f8f9fa;
                border-radius:12px;
                padding:1.2rem 1.5rem;
                border-left:5px solid {colour};
                margin-bottom:0.5rem;
            ">
                <div style="font-size:0.8rem;color:#888;font-weight:600;
                            text-transform:uppercase;">Top Prediction</div>
                <div style="font-size:1.4rem;font-weight:700;
                            color:#1a1a2e;">{top_name}</div>
                <div style="font-size:1.1rem;font-weight:600;
                            color:{colour};">{top_conf:.1f}% confident</div>
            </div>
            """, unsafe_allow_html=True)

            if top_conf >= 80:
                st.success(label)
            elif top_conf >= 50:
                st.warning(label)
            else:
                st.error(label)

        st.markdown("### Top 3 Predictions")
        render_chart(predictions)

        st.markdown("### Full Breakdown")
        for rank, (name, conf) in enumerate(predictions, 1):
            c1, c2, c3, c4 = st.columns([0.5, 2.5, 4, 1])
            c1.markdown(f"**#{rank}**")
            c2.markdown(name)
            c3.progress(int(conf))
            c4.markdown(f"**{conf:.1f}%**")

    else:
        st.markdown("""
        <div style="
            text-align:center;
            padding:3rem 1rem;
            background:#f8f9fa;
            border-radius:12px;
            border:2px dashed #ddd;
        ">
            <div style="font-size:3rem;">📸</div>
            <div style="font-size:1.1rem;color:#666;margin-top:0.5rem;">
                Upload a food image to get started
            </div>
            <div style="font-size:0.85rem;color:#aaa;margin-top:0.3rem;">
                JPG, PNG or WEBP accepted
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center;padding:2rem 0 1rem;color:#aaa;font-size:0.8rem;">
            Nigerian Food Classifier · EfficientNetB0 · 18 Classes · 85.75% Accuracy
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
