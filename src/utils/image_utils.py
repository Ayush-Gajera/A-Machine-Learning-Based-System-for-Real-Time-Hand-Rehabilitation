"""
Image utilities for  Hand Rehabilitation System.
"""

import base64
from PIL import Image
import streamlit as st


def resize_image(image_path, width, height):
    """
    Resize image to specified dimensions.
    
    Args:
        image_path: Path to the image file
        width: Target width
        height: Target height
        
    Returns:
        PIL.Image: Resized image or None if error
    """
    try:
        img = Image.open(image_path)
        img = img.resize((width, height))
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def autoplay_video(video_path):
    """
    Create autoplay video HTML component for Streamlit.
    
    Args:
        video_path: Path to the video file
    """
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        base64_video = base64.b64encode(video_bytes).decode("utf-8")
        st.markdown(
            f"""
            <video autoplay loop width="700" height="350">
                <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
            </video>
            """,
            unsafe_allow_html=True,
        ) 