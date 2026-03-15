import pyrealsense2 as rs

# Try to initialize WITHOUT specific config first
pipeline = rs.pipeline()
try:
    # This will attempt to start with the camera's default profiles
    print("Connecting to camera...")
    pipeline.start()
    print("Success! Camera is detected and streaming.")

    # Get 1 frame to confirm
    frames = pipeline.wait_for_frames()
    print("Frame received!")

    pipeline.stop()
except Exception as e:
    print(f"Internal Error: {e}")
