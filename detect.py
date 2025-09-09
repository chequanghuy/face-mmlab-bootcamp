import os
import cv2
import time
import argparse
import numpy as np

import torch

from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.general import draw_detections
from utils.box_utils import decode, decode_landmarks, nms



def preprocess(original_image):
    # read image
    
    image = np.float32(original_image)
    img_height, img_width, _ = image.shape

    # normalize image
    image -= (104, 117, 123)
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0)  # 1CHW
    image = image.to("cpu")
    return image

@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks




def detect(model, image, params):
    # load configuration and device setup
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")
    device = torch.device("cpu")

    resize_factor = 1


    img_height, img_width, _ = image.shape

    image = preprocess(image)
    

    # forward pass
    loc, conf, landmarks = inference(model, image)

    # generate anchor boxes
    priorbox = PriorBox(cfg, image_size=(img_height, img_width))
    priors = priorbox.generate_anchors().to(device)

    # decode boxes and landmarks
    boxes = decode(loc, priors, cfg['variance'])
    landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

    # scale adjustments
    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

    landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
    landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

    scores = conf.cpu().numpy()[:, 1]

    # filter by confidence threshold
    inds = scores > params.conf_threshold
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    # sort by scores
    order = scores.argsort()[::-1][:params.pre_nms_topk]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # apply NMS
    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(detections, params.nms_threshold)

    detections = detections[keep]
    landmarks = landmarks[keep]

    # keep top-k detections and landmarks
    detections = detections[:params.post_nms_topk]
    landmarks = landmarks[:params.post_nms_topk]

    # concatenate detections and landmarks
    detections = np.concatenate((detections, landmarks), axis=1)

    return detections
class InferenceConfig:
    """
    A class to hold inference configuration for the RetinaFace model.
    """
    def __init__(self):
        self.weights = 'weights/retinaface_mv1_0.25.pth'
        self.network = 'mobilenetv1_0.25'   #mobilenetv2
        self.conf_threshold = 0.02
        self.pre_nms_topk = 5000
        self.nms_threshold = 0.4
        self.post_nms_topk = 750
        self.vis_threshold = 0.8
        self.image_path = './assets/largest_selfie.jpg'


if __name__ == '__main__':
    params = InferenceConfig()
        
    model = RetinaFace(cfg=get_config(params.network)).to(torch.device("cpu"))
    model.eval()
    state_dict = torch.load(params.weights, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)

    original_image = cv2.imread('assets/large_selfi.jpg', cv2.IMREAD_COLOR)
    detections = detect(model, original_image, params)
    draw_detections(original_image, detections, params.vis_threshold)
    cv2.imwrite("face.jpg",original_image)