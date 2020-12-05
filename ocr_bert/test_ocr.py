# %load test_original.py
import copy
import textwrap
import os, os.path
import tensorflow as tf
import multiprocessing

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ocr.text_recognition import TextRecognition
from ocr.text_detection import TextDetection

from ocr.util import *
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry.polygon import orient
from skimage import draw
import time

UPLOAD_FOLDER = './uploads'
IMAGE_FOLDER = './image'
VIDEO_FOLDER = r'./video'
FOND_PATH = './ocr/STXINWEI.TTF'

dir_path = os.getcwd()

def init_ocr_model():

    detection_pb = dir_path+'/ocr/checkpoint/ICDAR_0.7.pb' # './checkpoint/ICDAR_0.7.pb'
    recognition_pb = dir_path+'/ocr/checkpoint/text_recognition_5435.pb'

    with tf.device('/cpu:0'):
        tf_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                   allow_soft_placement=True)

        detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
        recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
    label_dict = np.load(dir_path+'/ocr/reverse_label_dict_with_rects.npy',allow_pickle=True)[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
    return detection_model, recognition_model, label_dict



def draw_annotation(image, points, label, horizon=True, vis_color=(30,255,255)):#(30,255,255)
    points = np.asarray(points)
    points = np.reshape(points, [-1, 2])
    cv2.polylines(image, np.int32([points]), 1, (0, 255, 0), 2)

    image = Image.fromarray(image)
    width, height = image.size
    fond_size = int(max(height, width)*0.03)
    FONT = ImageFont.truetype(FOND_PATH, fond_size, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    # points = order_points(points)
    if horizon:
        DRAW.text((points[0][0], max(points[0][1] - fond_size, 0)), label, vis_color, font=FONT)
    else:
        lines = textwrap.wrap(label, width=1)
        y_text = points[0][1]
        for line in lines:
            width, height = FONT.getsize(line)
            DRAW.text((max(points[0][0] - fond_size, 0), y_text), line, vis_color, font=FONT)
            y_text += height
    image = np.array(image)
    return image



def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask



def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox



def detection(img_path, detection_model, recognition_model, label_dict, it_is_video=False):
    bgr_image = cv2.imread(img_path)
    print(bgr_image.shape) 
    r_list = []

    vis_image = copy.deepcopy(bgr_image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    r_boxes, polygons, scores = detection_model.predict(bgr_image)

    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = rgb_image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height >= width:
            scale = test_size / height
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            # print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                              right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.
        else:
            scale = test_size / width
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            # print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                              right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.

        image_padded = np.expand_dims(image_padded, 0)
        # print(image_padded.shape)

        results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
#         print(''.join(results))
        # print(probs)

        ccw_polygon = orient(Polygon(polygon.tolist()).simplify(5, preserve_topology=True), sign=1.0)
        pts = list(ccw_polygon.exterior.coords)[:-1]
        r_list.append(results)
        vis_image = draw_annotation(vis_image, pts, ''.join(results))
        # if height >= width:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results), False)
        # else:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results))


    return vis_image,r_list


def ocr_img(img_path,save_path, ocr_detection_model, ocr_recognition_model, ocr_label_dict):
    image,r_list = detection(img_path,ocr_detection_model, ocr_recognition_model, ocr_label_dict)
    cv2.imwrite(save_path, image)
    return r_list


