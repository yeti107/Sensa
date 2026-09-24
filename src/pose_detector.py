import mediapipe as mp
from pathlib import Path
import cv2

mpath = str(Path("models/pose_landmarker_full.task"))


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions  
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult 
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
     model_asset_path=mpath,
     delegate=BaseOptions.Delegate.CPU
    ),
running_mode=VisionRunningMode.IMAGE
)
with PoseLandmarker.create_from_options(options) as landmarker:   
 mp_image = mp.Image.create_from_file('test_images/4F9E9F40-6C05-4A3E-A402-3C35D9CD9038.jpeg')
 pose_landmarker_result = landmarker.detect(mp_image)

print (pose_landmarker_result)

