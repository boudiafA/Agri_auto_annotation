import sys
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

sys.path.insert(0, 'models/grit_src/third_party/CenterNet2/projects/CenterNet2/')

from centernet.config import add_centernet_config
from models.grit_src.grit.config import add_grit_config

from models.grit_src.grit.predictor import VisualizationDemo

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
# from utils.util import resize_long_edge_cv2

import cv2
def resize_long_edge_cv2(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    if scale != 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return img




# constants
WINDOW_NAME = "GRiT"


def dense_pred_to_caption(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ""
    for i in range(len(object_description)):
        new_caption += (object_description[i] + ": " + str(
            [int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
    return new_caption


def dense_pred_dict(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    scores = predictions["instances"].scores if predictions["instances"].has("scores") else None
    object_description = predictions["instances"].pred_object_descriptions.data

    prediction_list = []
    for i in range(len(object_description)):
        bbox = [round(float(a), 2) for a in boxes[i].tensor.cpu().detach().numpy()[0]]
        score = round(float(scores[i]), 2) if scores is not None else None

        prediction_dict = {
            'bbox': bbox,
            'score': score,
            'description': object_description[i],
        }
        prediction_list.append(prediction_dict)

    return prediction_list


def setup_cfg(args):
    cfg = get_cfg()
    if args["cpu"]:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    if args["test_task"]:
        cfg.MODEL.TEST_TASK = args["test_task"]
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser(device, checkpoint_path="grit_b_densecap_objectdet.pth"):
    arg_dict = {'config_file': "models/grit_src/configs/GRiT_B_DenseCap_ObjectDet.yaml", 'cpu': False,
                'confidence_threshold': 0.7, 'test_task': 'DenseCap',
                'opts': ["MODEL.WEIGHTS", checkpoint_path]}
    if device == "cpu":
        arg_dict["cpu"] = True
    return arg_dict


def image_caption_api(image_src, device, checkpoint_path="grit_b_densecap_objectdet.pth"):
    args2 = get_parser(device, checkpoint_path)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image_src:
        img = read_image(image_src, format="BGR")
        img = resize_long_edge_cv2(img, 384)
        predictions, visualized_output = demo.run_on_image(img)
        new_caption = dense_pred_to_caption(predictions)
    return new_caption


def image_caption_dict(image_src, device, checkpoint_path="grit_b_densecap_objectdet.pth"):
    args2 = get_parser(device, checkpoint_path)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image_src:
        img = read_image(image_src, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        dense_pred = dense_pred_dict(predictions)
    return dense_pred


def setup_grit(device, checkpoint_path="grit_b_densecap_objectdet.pth"):
    args2 = get_parser(device, checkpoint_path)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    return demo