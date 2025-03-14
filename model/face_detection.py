import cv2
import numpy as np
from tqdm import tqdm

def detect_faces(frames, min_face_size=30):
    """
    Detect faces in frames using OpenCV's Haar Cascade classifier instead of dlib
    
    Args:
        frames: List of frames to detect faces in
        min_face_size: Minimum face size to detect
        
    Returns:
        Tuple of (face_frames, count) where face_frames is a numpy array of detected faces
        and count is the number of faces detected
    """
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize array to store faces
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    
    # Process each frame
    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            if count < len(frames):
                # Extract and resize face
                face_image = frame[y:y+h, x:x+w]
                face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
                
                # Store face
                temp_face[count] = face_image
                count += 1
            else:
                break
    
    return ([], 0) if count == 0 else (temp_face[:count], count)