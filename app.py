import streamlit as st
import os
import tempfile
import shutil
import torch
from huggingface_hub import hf_hub_download
import cv2
from PIL import Image
import numpy as np
import time
import sys
import json
import graphviz
import pandas as pd
from datetime import datetime

# Add a custom path for model imports
if "model" not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your prediction functions
from model.pred_func import (
    load_genconvit,
    df_face,
    pred_vid,
    real_or_fake,
    set_result,
    store_result
)
from model.config import load_config

# Set page config
st.set_page_config(
    page_title="Deepfake Detection with GenConViT",
    page_icon="🎭",
    layout="wide"
)

# Initialize logs in session state
if 'logs' not in st.session_state:
    st.session_state.logs = []

def add_log(message):
    """Add a log entry with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

@st.cache_resource
def load_model_from_huggingface(model_type="both"):
    """Load the model weights from Hugging Face Hub based on selection"""
    config = load_config()
    add_log("Starting model weights download from Hugging Face Hub")
    
    os.makedirs("weight", exist_ok=True)
    
    with st.spinner("Downloading model weights from Hugging Face Hub..."):
        ed_path = hf_hub_download(
            repo_id="Deressa/GenConViT",
            filename="genconvit_ed_inference.pth",
        )
        vae_path = hf_hub_download(
            repo_id="Deressa/GenConViT",
            filename="genconvit_vae_inference.pth",
        )
        
        shutil.copy(ed_path, "weight/genconvit_ed_inference.pth")
        shutil.copy(vae_path, "weight/genconvit_vae_inference.pth")
    add_log("Model weights downloaded successfully")

    with st.spinner("Loading model..."):
        if model_type == "ed":
            model = load_genconvit(
                config,
                "genconvit",
                "genconvit_ed_inference",
                None,
                fp16=False
            )
            add_log("Loaded ED Model only")
        elif model_type == "vae":
            model = load_genconvit(
                config,
                "genconvit",
                None,
                "genconvit_vae_inference",
                fp16=False
            )
            add_log("Loaded VAE Model only")
        else:
            model = load_genconvit(
                config,
                "genconvit",
                "genconvit_ed_inference",
                "genconvit_vae_inference",
                fp16=False
            )
            add_log("Loaded both ED and VAE Models")

    return model, config

def is_video(file):
    """Check if a file is a valid video file"""
    try:
        cap = cv2.VideoCapture(file)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret
    except:
        return False

def create_flowchart(stage=None):
    """Creates a flowchart of the deepfake detection pipeline."""
    graph = graphviz.Digraph('pipeline', graph_attr={'rankdir': 'LR', 'size': '10,15'})

    stages = {
        "upload": {"label": "Upload\nVideo", "fillcolor": "#ddeedd", "color": "#336633", "done": False},
        "frames": {"label": "Extract\nFrames", "fillcolor": "#eef2ff", "color": "#336699", "done": False},
        "preprocessing": {"label": "Preprocess\nFrames", "fillcolor": "#fff0ee", "color": "#996633", "done": False},
        "model": {"label": "GenConViT\nModel", "fillcolor": "#f0e68c", "color": "#a67d3d", "done": False},
        "results": {"label": "Results", "fillcolor": "#c0c0c0", "color": "#555555", "done": False},
    }

    if stage:
        for key in stages:
            if key == stage:
                stages[key]["fillcolor"] = "#ffcc00"
                stages[key]["color"] = "#b8860b"
                break
            else:
                stages[key]["fillcolor"] = "#90ee90"
                stages[key]["color"] = "#006400"
                stages[key]["done"] = True

    for key, details in stages.items():
        graph.node(key, details["label"], fillcolor=details["fillcolor"], color=details["color"], shape='box', style='filled,rounded')

    graph.edge("upload", "frames")
    graph.edge("frames", "preprocessing")
    graph.edge("preprocessing", "model")
    graph.edge("model", "results")

    return graph

def extract_faces_from_frames(video_path, num_frames=15):
    """Extract faces from video frames and display some of them"""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = min(num_frames, total_frames)
    interval = max(1, total_frames // frames_to_extract)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_frames = []
    
    for i in range(0, total_frames, interval):
        if len(face_frames) >= frames_to_extract:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            face_frames.append(frame)
    
    cap.release()
    return face_frames[:frames_to_extract]

def process_video(video_file, model, config, num_frames=15, progress_bar=None, flowchart_placeholder=None):
    """Process a video file and return prediction"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_file_path = tmp_file.name

    total_steps = 4
    progress_step = 0

    try:
        add_log(f"Processing video: {video_file.name}")
        if flowchart_placeholder:
            flowchart_placeholder.graphviz_chart(create_flowchart("frames"))

        progress_step += 1
        if progress_bar:
            progress_bar.progress(progress_step / total_steps, "Extracting faces...")
            
        with st.spinner("Extracting faces from video frames..."):
            df = df_face(tmp_file_path, num_frames, "genconvit")
            add_log(f"Extracted {len(df)} face frames")

        if len(df) >= 1:
            if flowchart_placeholder:
                flowchart_placeholder.graphviz_chart(create_flowchart("preprocessing"))

            progress_step += 1
            if progress_bar:
                progress_bar.progress(progress_step / total_steps, "Preprocessing frames...")
            time.sleep(0.5)

            if flowchart_placeholder:
                flowchart_placeholder.graphviz_chart(create_flowchart("model"))

            progress_step += 1
            if progress_bar:
                progress_bar.progress(progress_step / total_steps, "Analyzing with GenConViT...")

            with st.spinner("Analyzing video..."):
                y, y_val = pred_vid(df, model)
                prediction = real_or_fake(y)
                confidence = float(y_val)
                add_log(f"Prediction: {prediction} with confidence {confidence:.4f}")
        else:
            prediction = "Unable to detect faces"
            confidence = 0.0
            add_log("No faces detected in video")

        if flowchart_placeholder:
            flowchart_placeholder.graphviz_chart(create_flowchart("results"))
        progress_step += 1
        if progress_bar:
            progress_bar.progress(progress_step / total_steps, "Results ready!")

        os.unlink(tmp_file_path)
        add_log("Temporary video file removed")
        return prediction, confidence, df

    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        add_log(f"Error processing video: {str(e)}")
        st.error(f"Error processing video: {str(e)}")
        return "Error", 0.0, None

def main():
    st.sidebar.title("GenConViT Deepfake Detector")
    page = st.sidebar.radio("Navigation", ["Home", "About", "How It Works"])
    
    model_type = st.sidebar.selectbox(
        "Select Model",
        options=["Both (ED + VAE)", "ED Model Only", "VAE Model Only"],
        index=0,
        help="Choose which model components to use for detection."
    )
    
    model_type_map = {
        "Both (ED + VAE)": "both",
        "ED Model Only": "ed",
        "VAE Model Only": "vae"
    }
    selected_model_type = model_type_map[model_type]

    if page == "Home":
        st.title("🎭 Deepfake Detection with GenConViT")
        st.markdown("""
        Upload a video to detect if it's a real or fake (manipulated) facial video.
        This app uses the GenConViT model to analyze facial videos for signs of manipulation.
        """)

        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        
        if not st.session_state.model_loaded:
            try:
                with st.spinner("⏳ Loading AI model..."):
                    model, config = load_model_from_huggingface(model_type=selected_model_type)
                st.success("✅ Model loaded successfully")
                st.session_state.model = model
                st.session_state.config = config
                st.session_state.model_loaded = True
                st.session_state.model_type = model_type
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                st.stop()
        else:
            model = st.session_state.model
            config = st.session_state.config

        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "wmv"])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            num_frames = st.slider("Number of frames to process", min_value=5, max_value=30, value=15)
        

        progress_bar_placeholder = st.empty()
        flowchart_placeholder = st.empty()
        
        result_container = st.container()
        details_container = st.container()
        
        if uploaded_file is not None:
            flowchart_placeholder.graphviz_chart(create_flowchart("upload"))
            progress_bar = progress_bar_placeholder.progress(0, "Starting analysis...")
            st.video(uploaded_file)
            
            prediction, confidence, tensor_data = process_video(
                uploaded_file, model, config, num_frames, progress_bar, flowchart_placeholder
            )
            
            with result_container:
                st.subheader("Analysis Results")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == "FAKE":
                        st.error("⚠️ DEEPFAKE DETECTED")
                        st.metric("Confidence", f"{confidence:.2f}")
                        st.markdown("This video appears to be manipulated.")
                    elif prediction == "REAL":
                        st.success("✅ AUTHENTIC VIDEO")
                        st.metric("Confidence", f"{(1 - confidence):.2f}")  # Show "real" confidence
                        st.markdown("This video appears to be authentic.")
                    else:
                        st.warning(f"⚠️ {prediction}")
                
                with col2:
                    if prediction != "Unable to detect faces" and prediction != "Error":
                        fake_percentage = confidence * 100
                        real_percentage = (1 - confidence) * 100
                        chart_data = pd.DataFrame({
                            "Category": ["Real", "Fake"],
                            "Percentage": [real_percentage, fake_percentage]
                        })
                        st.bar_chart(chart_data.set_index("Category"))
            
            with details_container:
                st.subheader("Detailed Analysis")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                details = {
                    "Metric": ["Video", "Model Used", "Frames Analyzed", "Result", "Confidence", "Date/Time"],
                    "Value": [
                        uploaded_file.name, 
                        model_type,
                        num_frames, 
                        prediction, 
                        f"{confidence:.4f}", 
                        current_time
                    ]
                }
                
                df_details = pd.DataFrame(details)
                st.dataframe(df_details, use_container_width=True)
                
                csv = df_details.to_csv(index=False)
                st.download_button(
                    label="📊 Export Results as CSV",
                    data=csv,
                    file_name=f"deepfake_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        
        # Logs Section
        with st.expander("Processing Logs", expanded=False):
            st.subheader("Logs")
            if st.session_state.logs:
                log_text = "\n".join(st.session_state.logs)
                st.text_area("Log Output", value=log_text, height=200, disabled=True)
            else:
                st.info("No logs available yet.")
            if st.button("Clear Logs"):
                st.session_state.logs = []
                st.rerun()
    
    elif page == "About":
        st.title("About GenConViT")
        st.markdown("""
        ## What is GenConViT?
        
        GenConViT is a deepfake detection model that combines convolutional neural networks with vision transformers
        to detect manipulated facial videos with high accuracy.
        
        ### Key Features
        
        - **Robust Detection**: Trained on multiple deepfake datasets
        - **High Accuracy**: Achieves state-of-the-art performance
        - **Real-time Analysis**: Fast processing for quick results
        
        ### Capabilities
        
        The model can detect various types of facial manipulations including:
        - Face swaps
        - Face reenactment
        - Face synthesis
        - Attribute manipulation
        
        ### Model Architecture
        """)
        
        st.image("pipeline_architecture.png", 
                 caption="GenConViT Architecture Diagram")
        
        st.markdown("""
        ### Citations
        
        If you use GenConViT in your research or applications, please cite:
        
        ```
        @article{deressa2023genconvit,
          title={GenConViT: Generalized Convolutional Vision Transformer for Deepfake Detection},
          author={Deressa, Safal and Colleagues},
          journal={arXiv preprint},
          year={2023}
        }
        ```
        
        ### Source Code
        
        The model is available on GitHub: [https://github.com/Deressa/GenConViT](https://github.com/Deressa/GenConViT)
        """)
        
    elif page == "How It Works":
        st.title("How GenConViT Works")
        st.markdown("""
        ## Deepfake Detection Pipeline
        
        GenConViT processes videos through a series of steps to determine if they're real or fake:
        """)
        st.graphviz_chart(create_flowchart())
        st.markdown("""
        ### 1. Video Upload
        The process begins when you upload a video file to be analyzed.
        
        ### 2. Frame Extraction
        The system extracts key frames from the video for analysis.
        
        ### 3. Preprocessing
        Frames are preprocessed to detect and crop faces, normalize lighting, and prepare for analysis.
        
        ### 4. Model Analysis
        The GenConViT model analyzes the facial features and movement patterns to detect signs of manipulation.
        
        ### 5. Results
        The system provides a prediction along with a confidence score, indicating whether the video is real or fake.
        
        ## Technical Details
        
        GenConViT combines the strengths of:
        - Convolutional Neural Networks (CNN) for local feature extraction
        - Vision Transformers (ViT) for global context understanding
        
        This hybrid approach enables better detection across different types of deepfakes and manipulation techniques.
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 GenConViT")
st.sidebar.markdown("Created by Safal Immanuel Sabari")

if __name__ == "__main__":
    main()