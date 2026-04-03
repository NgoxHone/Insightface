#!/usr/bin/env python3
"""Test tracking stream endpoint"""
import requests
import cv2
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_tracking_stream():
    """Fetch a few frames from the tracking stream"""
    url = "http://localhost:5001/api/tracking/stream"
    params = {
        "source": "0",  # webcam
        "resolution": "640x480",
        "frame_skip": "1",
        "recognition_interval": "2.0"
    }

    print(f"Connecting to {url}...")
    try:
        response = requests.get(url, params=params, stream=True, timeout=10)
        response.raise_for_status()

        boundary = response.headers.get('Content-Type', '').split('boundary=')[-1]
        if not boundary:
            print("Not an MJPEG stream?")
            return

        print(f"Stream boundary: {boundary}")
        print("Reading frames...")

        bytes_buffer = b''
        frame_count = 0
        max_frames = 5

        for chunk in response.iter_content(chunk_size=1024):
            bytes_buffer += chunk
            # Look for JPEG end marker
            while b'\xff\xd9' in bytes_buffer:
                jpeg_end = bytes_buffer.find(b'\xff\xd9') + 2
                jpeg_data = bytes_buffer[:jpeg_end]
                bytes_buffer = bytes_buffer[jpeg_end:]

                # Decode JPEG
                img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    frame_count += 1
                    h, w = img.shape[:2]
                    print(f"Frame {frame_count}: {w}x{h}")

                    # Save a sample frame
                    if frame_count == 1:
                        cv2.imwrite('/tmp/tracking_test_frame.jpg', img)
                        print("Saved sample frame to /tmp/tracking_test_frame.jpg")

                    if frame_count >= max_frames:
                        break

            if frame_count >= max_frames:
                break

        print(f"\n✅ Successfully received {frame_count} frames from tracking stream!")
        print("Check /tmp/tracking_test_frame.jpg for a sample")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the backend running on http://localhost:5001?")
        print("   Start with: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_tracking_stream()
