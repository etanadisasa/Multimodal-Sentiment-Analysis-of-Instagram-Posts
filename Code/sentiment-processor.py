# ________________________________________________________________________
# FILE NAME: sentiment-processor.py
# AUTHOR: Etana Disasa
# EMAIL: etanan@gmail.com
# DATE: August 22, 2025
# TITLE: Multimodal Sentiment Analysis of Instagram Posts
# _________________________________________________________________


import streamlit as st
from socialmedia_utils import (
    analyze_image_emotion,
    analyze_text_sentiment,
    calculate_cumulative_emotion,   
    extract_video_frame,
    generate_image_caption,
    get_instagram_post_data,
    get_shortcode_from_url,
    load_face_emotion_model,
    load_image_detection_model,
    load_image_with_fallback,
    load_text_sentiment_model,   
    summarize_sentiments,
)
import plotly.express as px
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from transformers import pipeline
import os

# Disable tqdm interactive features to suppress IProgress warning
# This avoids cluttering the Streamlit output when using tqdm in model loading or processing
os.environ["TQDM_DISABLE"] = "1"

# Load and cache the text sentiment analysis model
# Cached to avoid reloading on every interaction, improving app performance
@st.cache_resource(show_spinner=False)
def get_text_model():
    return load_text_sentiment_model()

# Load and cache the image detection model (object detection, scene recognition)
@st.cache_resource(show_spinner=False)
def get_image_model():
    return load_image_detection_model()

# Load and cache the face emotion detection model
# Used to detect human faces and predict emotional expression
@st.cache_resource(show_spinner=False)
def get_face_model():
    return load_face_emotion_model()

# Load and cache the YOLO model for object detection in images
# Handles exceptions to provide user feedback if the model file is missing
@st.cache_resource(show_spinner=False)
def get_yolo_model():
    try:
        return YOLO("yolov8n.pt")
    except FileNotFoundError:
        st.error("YOLO model file 'yolov8n.pt' not found. Please ensure it is in the working directory.")
        return None
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Load NSFW detection model for image moderation
# Tries local model first, falls back to online model; warns if model cannot be loaded
@st.cache_resource(show_spinner=False)
def get_nsfw_model():
    local_path = "./nsfw_model"
    try:
        if os.path.exists(local_path):
            return pipeline("image-classification", model=local_path)
        return pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    except Exception as e:
        st.warning(
            f"Failed to load NSFW detection model: {e}. Content moderation will be skipped. "
            "To enable offline mode, download the model using the provided 'download_nsfw_model.py' script "
            "and ensure './nsfw_model' exists in the app directory."
        )
        return None

if "session_id" not in st.session_state:
    st.session_state.session_id = None

def main():
    st.set_page_config(layout="wide", page_title="Instagram Sentiment Analyzer")
    
    # Header
    st.title("🎭 Instagram Post Sentiment Analyzer")
    st.markdown("*Analyze emotions/sentiments in Instagram posts through media, captions, and comments using AI*")
    
    # Load models
    text_model = get_text_model()
    _, image_extractor = get_image_model()
    face_model, image_extractor = get_face_model()
    yolo_model = get_yolo_model()
    nsfw_model = get_nsfw_model()

    if yolo_model is None:
        st.error("Cannot proceed without a valid YOLO model.")
        return

    if nsfw_model is None:
        st.info("Content moderation unavailable. All media will be processed.")

    # Input Section
    url = st.text_input("📎 **Enter Instagram Post URL:**", placeholder="https://www.instagram.com/p/...")
    username = "studentregis"

    if st.button("🔍 **Analyze Post**", type="primary"):
        if not url.strip():
            st.warning("Please enter a valid Instagram post URL.")
            return
            
        # Fetch data
        shortcode = get_shortcode_from_url(url) or "unknown"
        with st.spinner("🔄 Fetching Instagram post data..."):
            try:
                data = get_instagram_post_data(url, username, session_id=st.session_state.session_id)
            except Exception as e:
                st.error(f"Unexpected error during extraction: {e}")
                return

        if "error" in data:
            st.error(f"❌ {data['error']}")
            return

        # Initialize tracking variables
        all_emotions = []
        all_faces = []
        media_emotions = []
        
        # ========== MAIN ANALYSIS SECTIONS ==========
        
        # 1. POST OVERVIEW
        st.markdown("---")
        st.header("📋 Post Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Media Items", len(data.get("media_urls", [])))
        with col2:
            st.metric("Comments", len(data.get("top_comments", [])))
        with col3:
            st.metric("Likes", data.get("likes", 0))
        with col4:
            media_types_str = ", ".join(mt.capitalize() for mt in set(data.get("media_types", [])))
            st.metric("Media Type", media_types_str if media_types_str else "None")

        # Caption Preview
        caption = data.get("caption", "")
        if caption:
            st.markdown("**📝 Caption:**")
            short_caption = caption[:200] + "..." if len(caption) > 200 else caption
            st.markdown(
                f"""
                <div style="padding:10px; border-radius:12px; background-color:#f0f2f6;">
                    <span style="font-size:14px; color:#333;">
                        {short_caption}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # 2. MEDIA ANALYSIS
        st.markdown("---")
        st.header("🎨 Media Analysis")
        
        media_urls = data.get("media_urls", [])
        media_types = data.get("media_types", [])

        if media_urls:
            main_cols = st.columns(2)  # ✅ 2 main columns across

            for i, (media_url, media_type) in enumerate(zip(media_urls, media_types)):
                with main_cols[i % 2]:
                    subcol_media, subcol_analysis = st.columns([1, 2])  # ✅ media | analysis

                    # Load media
                    img = None
                    if media_type == "image":
                        img = load_image_with_fallback(media_url)
                    elif media_type == "video":
                        img = extract_video_frame(media_url, frame_position=0.0)

                    if img:
                        # NSFW Check
                        if nsfw_model:
                            nsfw_result = nsfw_model(img)
                            is_nsfw = any(r["label"] == "nsfw" and r["score"] > 0.7 for r in nsfw_result)
                            if is_nsfw:
                                st.warning("⚠️ Content skipped due to inappropriate content detection")
                                continue

                        with subcol_media:
                            st.image(img, caption=f"{media_type.title()}-{shortcode}-{i+1}".upper(), use_container_width=True)

                        with subcol_analysis:
                            # Image Description & Sentiment
                            image_description = generate_image_caption(media_url if media_type == "image" else img)
                            image_emotion = analyze_text_sentiment(text_model, image_description)
                            image_emotion["label"] = image_emotion["label"].upper()

                            st.markdown(f"**{media_type.title()}-{shortcode}-{i+1}**".upper())
                            st.write(f"**AI Generated Description:** {image_description.capitalize()}")
                            st.write(f"**Sentiment:** {image_emotion['label']} ({image_emotion['score']:.2f} confidence)")

                            # Face Detection
                            yolo_results = yolo_model.predict(img)
                            face_emotions = []
                            faces_detected = 0

                            MIN_CONFIDENCE = 0.75
                            MIN_FACE_PIXELS = 30

                            for face in yolo_results:
                                if hasattr(face, 'boxes') and len(face.boxes) > 0:
                                    for box, score in zip(face.boxes.xyxy, face.boxes.conf):
                                        if score < MIN_CONFIDENCE:
                                            continue
                                        x1, y1, x2, y2 = map(int, box)
                                        width, height = x2 - x1, y2 - y1
                                        if width < MIN_FACE_PIXELS or height < MIN_FACE_PIXELS:
                                            continue

                                        face_img = img.crop((x1, y1, x2, y2))
                                        face_emotion = analyze_image_emotion(face_model, image_extractor, face_img)
                                        face_emotion["label"] = face_emotion["label"].upper()
                                        face_emotions.append(face_emotion)
                                        faces_detected += 1

                                        all_faces.append({
                                            "image_idx": i,
                                            "source_type": media_type,
                                            "emotion_label": face_emotion["label"],
                                            "emotion_score": face_emotion["score"],
                                            "thumb": face_img.copy().resize((160, 160))
                                        })

                            # Face Analysis Summary
                            st.markdown("**👤 Face Analysis:**")
                            if face_emotions:
                                st.write(f"Faces detected: **{faces_detected}**")
                                emotion_counts = Counter([fe["label"] for fe in face_emotions])
                                for emotion, count in emotion_counts.most_common():
                                    st.write(f"- {emotion}: {count} face(s)")

                                # Determine final media emotion
                                weighted_emotion, top_emotion = calculate_cumulative_emotion(all_faces[-faces_detected:])
                                final_emotion = {"label": top_emotion.upper(), "score": max(weighted_emotion.values())}
                                st.success(f"**Overall {media_type.capitalize()} Emotion from Face Analysis: {final_emotion['label']} ({final_emotion['score']:.2f})**")
                            else:
                                st.write("No faces detected")
                                final_emotion = image_emotion
                                st.success(f"**{media_type.capitalize()} Sentiment from AI Generated Description: {final_emotion['label']} (Confidence: {final_emotion['score']:.2f})**")
                            media_emotions.append(final_emotion)
                            all_emotions.append(final_emotion)
        else:
            st.info("No images or thumbnails available for this post.")
        
        # 3. DETAILED FACE GALLERY (Collapsible)
        if all_faces:        
            st.markdown("---")
            st.subheader("**Detected Face Images**")
            with st.expander("View All Detected Faces", expanded=False):
                st.info("All faces detected across the post (privacy-sensitive content)")
                
                # Group by media
                faces_by_media = {}
                for face in all_faces:
                    faces_by_media.setdefault(face["image_idx"], []).append(face)
                
                for media_idx, faces in faces_by_media.items():
                    st.markdown(f"**{media_types[media_idx].title().capitalize()}-{shortcode}-{media_idx + 1}** - {len(faces)} face(s)")
                    cols = st.columns(min(4, len(faces)))
                    for i, face in enumerate(faces):
                        with cols[i % 4]:
                            st.image(face["thumb"], caption=f"{face['emotion_label']}\n({face['emotion_score']:.2f})")

        st.markdown("---")

        # 4. TEXT ANALYSIS
        st.header("📝 Text Analysis")
        
        # Caption Analysis
        if caption:
            caption_emotion = analyze_text_sentiment(text_model, caption)
            caption_emotion["label"] = caption_emotion["label"].upper()
            all_emotions.append(caption_emotion)
            
            st.subheader("Caption Sentiment")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(caption)
            with col2:
                st.metric("Sentiment", caption_emotion["label"], f"{caption_emotion['score']:.2f}")
        
        # Comments Analysis
        top_comments = data.get("top_comments", [])
        if top_comments:
            st.subheader("Comments Analysis")
            comment_emotions = []
            
            for i, c in enumerate(top_comments, 1):
                comment_emotion = analyze_text_sentiment(text_model, c['text'])
                comment_emotion["label"] = comment_emotion["label"].upper()
                comment_emotions.append(comment_emotion)
                all_emotions.append(comment_emotion)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{c['username']}**: {c['text']}")
                with col2:
                    st.write(f"**{comment_emotion['label']}** ({comment_emotion['score']:.2f})")
        else:
            st.info("No comments available for analysis")
        


        # 5. SUMMARY DASHBOARD
        st.markdown("---")
        st.header("📊 Overall Post Emotion/Sentiment Summary")
        
        if all_emotions:
            # Build source_types list to track the source of each emotion
            source_types = []
            for em in media_emotions:
                source_types.append("media")
            if caption_emotion:
                source_types.append("caption")
            for em in comment_emotions:
                source_types.append("comment")
            
            # Call summarize_sentiments with source types for 50/30/20 weighting
            summary = summarize_sentiments(all_emotions, source_types=source_types)            
            # Key Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dominant Emotion", summary['most_common_label'], f"{summary['most_common_count']:.2f} Weighted Score")
            with col2:
                st.metric("Average Confidence", f"{summary['average_score']:.2f}")
            with col3:
                st.metric("Total Items Analyzed", len(all_emotions))
            

            # Visualizations
            col1, col2 = st.columns(2)

            emotion_colors = {
                "POSITIVE": "#67C28D",   # green
                "NEGATIVE": "#a01000",   # red
                "NEUTRAL": "#5C5C5C",    # gray
            }
        
            with col1:
                st.subheader("Emotion Distribution")
                fig = px.bar(
                    x=list(summary['distribution'].keys()),
                    y=list(summary['distribution'].values()),
                    title="Emotion Frequency",
                    color=list(summary['distribution'].keys()),
                    color_discrete_map=emotion_colors
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            

            with col2:
                st.subheader("Emotion Breakdown")
                fig = px.pie(
                    names=list(summary['distribution'].keys()),
                    values=list(summary['distribution'].values()),
                    title="Overall Emotion Distribution",
                    color=list(summary['distribution'].keys()),
                    color_discrete_map=emotion_colors
                )
                st.plotly_chart(fig, use_container_width=True)
        
if __name__ == "__main__":
    main()