# ________________________________________________________________________
# FILE NAME: socialmedia_utils.py
# AUTHOR: Etana Disasa
# EMAIL: etanan@gmail.com
# DATE: August 22, 2025
# TITLE: Multimodal Sentiment Analysis of Instagram Posts
#_________________________________________________________________________

"""
Utility functions for Instagram data extraction, sentiment analysis,
image processing, video frame extraction, and session management.
"""

# -----------------------------
# Imports
# -----------------------------
from collections import defaultdict
from io import BytesIO
from urllib.parse import urlparse

import cv2
import emoji
import instaloader
import os
import requests
import torch
import uuid
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline
)
from ultralytics import YOLO


# -----------------------------
# Facial Detection & Sentiment Analysis
# -----------------------------

# Load the YOLO model once
yolo_model = YOLO("yolov8n.pt")  # small, fast model


# Initialize emotion classifier once
# Load the model
extractor = AutoFeatureExtractor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# Wrap in pipeline
face_emotion_classifier = pipeline("image-classification", model=model, feature_extractor=extractor)



# 🔄 Face Emotion to Sentiment Mapping
FACE_TO_SENTIMENT = {
    # GoEmotions (bert-base-go-emotion) labels
    "ADMIRATION": "POSITIVE",
    "AMUSEMENT": "POSITIVE",
    "APPROVAL": "POSITIVE",
    "CARING": "POSITIVE",
    "DESIRE": "POSITIVE",
    "EXCITEMENT": "POSITIVE",
    "GRATITUDE": "POSITIVE",
    "JOY": "POSITIVE",
    "LOVE": "POSITIVE",
    "OPTIMISM": "POSITIVE",
    "PRIDE": "POSITIVE",
    "RELIEF": "POSITIVE",
    "ANGER": "NEGATIVE",
    "ANNOYANCE": "NEGATIVE",
    "DISAPPOINTMENT": "NEGATIVE",
    "DISAPPROVAL": "NEGATIVE",
    "DISGUST": "NEGATIVE",
    "EMBARRASSMENT": "NEGATIVE",
    "FEAR": "NEGATIVE",
    "GRIEF": "NEGATIVE",
    "NERVOUSNESS": "NEGATIVE",
    "REMORSE": "NEGATIVE",
    "SADNESS": "NEGATIVE",
    "CONFUSION": "NEUTRAL",
    "CURIOSITY": "NEUTRAL",
    "REALIZATION": "NEUTRAL",
    "SURPRISE": "POSITIVE",  # Consistent with facial SURPRISED
    "NEUTRAL": "NEUTRAL",
    # Facial emotions (dima806/facial_emotions_image_detection)
    "HAPPY": "POSITIVE",
    "SURPRISED": "POSITIVE",
    "JOY": "POSITIVE",
    "ADMIRATION": "POSITIVE",
    "SAD": "NEGATIVE",
    "ANGRY": "NEGATIVE",
    "ANGER": "NEGATIVE",
    "FEAR": "NEGATIVE",
    "DISGUST": "NEGATIVE",
    "DISGUSTED": "NEGATIVE",
    "NEUTRAL": "NEUTRAL"
}

def normalize_emotion_label(label):
    return FACE_TO_SENTIMENT.get(label.upper(), label.upper())


# -----------------------------
# Image Analysis
# -----------------------------
_blip_processor = None
_blip_model = None
_image_emotion_model = None
_image_emotion_extractor = None



def analyze_image_emotion(model, extractor, img):
    inputs = extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    label_idx = scores.argmax().item()
    score = scores.max().item()
    label = model.config.id2label[label_idx]
    return {"label": label, "score": score}



def generate_image_caption(image_url_or_path):
    """Generate caption from image using BLIP."""
    processor, model = load_blip_model()

    # Load image
    if isinstance(image_url_or_path, str):
        if image_url_or_path.startswith("http"):
            image = Image.open(requests.get(image_url_or_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_url_or_path).convert("RGB")
    elif isinstance(image_url_or_path, Image.Image):
        image = image_url_or_path.convert("RGB")
    else:
        raise ValueError("Input must be a URL, file path, or PIL.Image object")

    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=50)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption

def load_blip_model():
    """Load BLIP image captioning model and processor."""
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor, _blip_model

def load_image_detection_model():
    """Load image classification model and feature extractor."""
    model_name = "google/vit-base-patch16-224"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

def load_face_emotion_model():
    """Load image emotion classifier model and extractor."""
    global _image_emotion_model, _image_emotion_extractor
    if _image_emotion_model is None or _image_emotion_extractor is None:
        model_name = "dima806/facial_emotions_image_detection"
        _image_emotion_model = AutoModelForImageClassification.from_pretrained(model_name)
        _image_emotion_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return _image_emotion_model, _image_emotion_extractor

def load_image_with_fallback(url):
    """Fetch image from URL and return PIL.Image or None."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and "image" in response.headers.get('Content-Type', ''):
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to fetch valid image: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# -----------------------------
# Text Analysis
# -----------------------------
def analyze_text_sentiment(model, text, top_n=1, min_score_threshold=0.01):
    """Analyze emotion in text using a pretrained classifier."""
    processed_text = preprocess_text(text)
    if not processed_text.strip():
        return {"label": "NEUTRAL", "score": 0.0}
    
    all_scores = model(processed_text)[0]  # list of dicts
    filtered_scores = [s for s in all_scores if s['score'] >= min_score_threshold]
    if not filtered_scores:
        filtered_scores = all_scores  # fallback
    
    filtered_scores.sort(key=lambda x: x['score'], reverse=True)
    top_emotions = [{"label": s['label'].upper(), "score": s['score']} for s in filtered_scores[:top_n]]
    
    return top_emotions[0] if top_n == 1 else top_emotions

def load_text_sentiment_model():
    """Load Hugging Face pipeline for text emotion analysis."""
    model_name = "bhadresh-savani/bert-base-go-emotion"
    return pipeline("text-classification", model=model_name, return_all_scores=True)

def preprocess_text(text):
    """Convert emojis in text to descriptive names."""
    return emoji.demojize(text)


def summarize_sentiments(emotion_list, source_types=None):
    """
    Summarize sentiments with 50% media, 30% caption, 20% comments weighting.
    
    Args:
        emotion_list: list of dicts like {'label': 'HAPPY', 'score': 0.91}
        source_types: list of strings indicating source ('media', 'caption', 'comment') for each emotion
                     If None, assumes equal weighting
    
    Returns:
        dict with most_common_label, most_common_count, average_score, distribution
    """
    if not emotion_list:
        return {
            "most_common_label": "NEUTRAL",
            "most_common_count": 0,
            "average_score": 0.0,
            "distribution": {}
        }

    dist = defaultdict(float)
    total_score = 0.0
    total_weight = 0.0

    # Define weights for each source type
    weights = {
        "media": 0.5,
        "caption": 0.3,
        "comment": 0.2
    }

    # If source_types is not provided, default to equal weighting
    if source_types is None:
        source_types = ["media"] * len(emotion_list)

    for i, (em, source) in enumerate(zip(emotion_list, source_types)):
        label = normalize_emotion_label(em["label"])
        score = em.get("score", 1.0)
        weight = weights.get(source, 1.0 / len(emotion_list))  # Fallback to equal weight if source unknown

        dist[label] += weight * score
        total_score += score * weight
        total_weight += weight

    # Normalize distribution
    normalized_dist = {k: v / total_weight for k, v in dist.items()} if total_weight > 0 else {}
    most_common_label = max(normalized_dist, key=normalized_dist.get) if normalized_dist else "NEUTRAL"
    most_common_count = normalized_dist.get(most_common_label, 0)

    return {
        "most_common_label": most_common_label,
        "most_common_count": most_common_count,
        "average_score": round(total_score / total_weight, 3) if total_weight > 0 else 0.0,
        "distribution": dict(normalized_dist)
    }
# -----------------------------
# Instagram Handling
# -----------------------------


def get_instagram_post_data(url, username, session_id=None):
    """Fetch Instagram post details, returning captions, media URLs, and top comments."""
    shortcode = get_shortcode_from_url(url)
    if not shortcode:
        return {"error": "Could not extract shortcode from URL"}
   
    L = instaloader.Instaloader()
    try:
        L.load_session_from_file(username)
    except FileNotFoundError:
        return {"error": f"Session file not found for username: {username}"}
   
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except Exception as e:
        return {"error": f"Could not fetch post: {str(e)}"}
   
    caption = post.caption or ""
    likes = post.likes
    comment_count = post.comments
    media_urls = []
    media_types = []
   
    # Debug info about the post
    post_info = {
        "typename": post.typename,
        "is_video": post.is_video,
        "shortcode": shortcode,
        "comment_count": comment_count
    }
   
    if post.typename == "GraphSidecar":
        for node in post.get_sidecar_nodes():
            media_urls.append(node.video_url if node.is_video else node.display_url)
            media_types.append("video" if node.is_video else "image")
    else:
        media_urls.append(post.video_url if post.is_video else post.url)
        media_types.append("video" if post.is_video else "image")
   
    top_comments = []
    comments_error = None
    
    try:
        print(f"[DEBUG] Attempting to fetch comments for {post.typename} post (shortcode: {shortcode})")
        print(f"[DEBUG] Post indicates {comment_count} comments available")
        
        comment_iterator = post.get_comments()
        comments_fetched = 0
        
        for comment in comment_iterator:
            if len(top_comments) >= 5:
                break
            top_comments.append({
                "username": comment.owner.username,
                "text": comment.text,
                "date": comment.created_at_utc.strftime("%Y-%m-%d"),
            })
            comments_fetched += 1
            print(f"[DEBUG] Fetched comment {comments_fetched} from {comment.owner.username}")
        
        print(f"[DEBUG] Successfully fetched {len(top_comments)} comments")
        
    except Exception as e:
        comments_error = str(e)
        print(f"[Comments Error] Could not fetch comments: {e}")
        print(f"[Comments Error] Error type: {type(e).__name__}")
        
        # Try to get more specific error information
        if hasattr(e, 'response'):
            print(f"[Comments Error] Response status: {getattr(e.response, 'status_code', 'N/A')}")
        
    return {
        "caption": caption,
        "likes": likes,
        "comment_count": comment_count,
        "media_urls": media_urls,
        "media_types": media_types,
        "top_comments": top_comments,
        "session_id": session_id,
        "post_info": post_info,  # Add debug info
        "comments_error": comments_error,  # Add error info if any
    }

def get_shortcode_from_url(url):
    """Extract shortcode from Instagram post URL."""
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] in ["p", "reel", "tv"]:
            return path_parts[1]
        elif len(path_parts) == 1:
            return path_parts[0]  # fallback
        return None
    except Exception as e:
        print(f"[URL Error] Could not extract shortcode: {e}")
        return None

# -----------------------------
# Post Processing
# -----------------------------


def calculate_cumulative_emotion(faces):
    """
    Compute cumulative emotions across all detected faces,
    factoring in confidence scores.
    Returns normalized distribution and top emotion.
    """
    emotion_totals = {}
    total_weight = 0.0

    for f in faces:
        label = f["emotion_label"]
        score = f["emotion_score"]

        emotion_totals[label] = emotion_totals.get(label, 0.0) + score
        total_weight += score

    # Normalize
    normalized = {}
    if total_weight > 0:
        normalized = {k: v / total_weight for k, v in emotion_totals.items()}

    # Get top emotion
    top_emotion = None
    if normalized:
        top_emotion = max(normalized.items(), key=lambda x: x[1])[0]

    return normalized, top_emotion

# -----------------------------
# Video Processing
# -----------------------------
def extract_video_frame(video_url, frame_position=0.3):
    """Download video and extract a frame at the specified relative position."""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        video_bytes = BytesIO(response.content)
        
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes.getbuffer())
        
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(total_frames * frame_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        cap.release()
        
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None
    except Exception as e:
        print(f"[extract_video_frame] Error: {e}")
        return None

