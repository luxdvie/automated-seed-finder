# Capture module
from .video_capture import VideoCapture
from .frame_processor import FrameProcessor
from .screen_capture import ScreenCapture, screen_capture

__all__ = ["VideoCapture", "FrameProcessor", "ScreenCapture", "screen_capture"]
