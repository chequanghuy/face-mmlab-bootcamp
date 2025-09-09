import cv2
import numpy as np
from skimage import transform as trans

def draw_detections(original_image, detections, vis_threshold):
    """
    Draws bounding boxes and landmarks on the image based on multiple detections.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of detected bounding boxes and landmarks.
        vis_threshold (float): The confidence threshold for displaying detections.
    """

    # Colors for visualization
    LANDMARK_COLORS = [
        (0, 0, 255),    # Right eye (Red)
        (0, 255, 255),  # Left eye (Yellow)
        (255, 0, 255),  # Nose (Magenta)
        (0, 255, 0),    # Right mouth (Green)
        (255, 0, 0)     # Left mouth (Blue)
    ]
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    # Filter by confidence
    detections = detections[detections[:, 4] >= vis_threshold]

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    landmarks = detections[:, 5:15].reshape(-1, 5, 2).astype(np.int32)

    for box, score, landmark in zip(boxes, scores, landmarks):
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

        # Draw confidence score
        text = f"{score:.2f}"
        cx, cy = box[0], box[1] + 12
        cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)

        # Draw landmarks
        for point, color in zip(landmark, LANDMARK_COLORS):
            cv2.circle(original_image, point, 1, color, 4)
def alignment(cv_img, dst, dst_w, dst_h):
    if dst_w == 112 and dst_h == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
    tform = trans.SimilarityTransform()
    # print(dst.shape, src.shape)
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
    return face_img
def crop_and_aligned(original_image, detections, vis_threshold):
    # Filter by confidence
    detections = detections[detections[:, 4] >= vis_threshold]

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    landmarks = detections[:, 5:15].reshape(-1, 5, 2).astype(np.int32)
    # print(landmarks.shape)
    faces = []
    for box, score, landmark in zip(boxes, scores, landmarks):
        h,w,_ = original_image.shape
        box[0] = max(0, min(box[0], w))
        box[2] = max(0, min(box[2], w))
        box[1] = max(0, min(box[1], h))
        box[3] = max(0, min(box[3], h))
        face = original_image[box[1]:box[3], box[0]:box[2]]
        if 0 in face.shape:
            continue
        # print((box[2] - box[0]),(box[3] - box[1]),"-----",(box[2] - box[0])*(box[3] - box[1]),box[1],box[3], box[0],box[2])
        face = cv2.resize(face, (112,112))
        ratio = [112/(box[2] - box[0]), 112/(box[3] - box[1])]
        for i, lm in enumerate(landmark):
            landmark[i][0] = (lm[0] - box[0]) * ratio[0]
            landmark[i][1] = (lm[1] - box[1]) * ratio[1]
        dst = np.array(landmark,dtype=np.float32)
        face_112x112 = alignment(face, dst, 112, 112)
        faces.append(face_112x112)
    return faces
