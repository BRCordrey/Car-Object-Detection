from io import BytesIO

import numpy as np
import cv2
import csv
import os

import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import tensorflow as tf
from utils import visualization_utils as vis_util, ops as utils_ops, label_map_util

category_index = label_map_util.create_category_index_from_labelmap('mscoco_label_map.pbtxt',
                                                                    use_display_name=True)
data = csv.writer(open('data.csv', 'w'))

model = tf.saved_model.load('saved_model')
matplotlib.use('TkAgg')


def run_inference_for_single_image(_model, image):
    image = np.asarray(image)

    input_tensor = tf.convert_to_tensor(image)

    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = _model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def getBox(dict):
    for index, score in enumerate(dict['detection_scores']):
        if dict['detection_classes'][index] == 3 and score > .5:
            label = category_index[dict['detection_classes'][index]]['name']
            ymin, xmin, ymax, xmax = dict['detection_boxes'][index]
            return box_cords_to_real_location(800, 600, ymin, xmin, ymax, xmax)


def box_cords_to_real_location(img_width, img_height, ymin, xmin, ymax, xmax):
    return [int(img_height * ymin), int(img_width * xmin), int(img_height * ymax), int(img_width * xmax)]


def image_to_np(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference(model):
    for image in os.listdir('acura-nsx-1991'):
        row = [image, 800, 600]
        image_np = image_to_np('acura-nsx-1991/' + image)
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        getBox(output_dict)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        image_np = cv2.resize(image_np, (800, 600))
        plt.imshow(image_np)
        plt.show()
        data.writerow(row)


run_inference(model)
