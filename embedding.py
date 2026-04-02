import insightface
import cv2
import numpy as np
 
#load facial recognition model (ArcFace)
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0)
 
#Read image
img = cv2.imread("face1.jpg")
 
#Face Detection
faces = model.get(img)
 
#If there is a face, get the embedding
if faces:
    face_embedding = faces[0].normed_embedding
    print(f"Embedding vector (512-D): {face_embedding}")