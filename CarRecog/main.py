import numpy as np
import cv2

import tensorflow as tf
from utils import visualization_utils as vis_util, ops as utils_ops, label_map_util

category_index = label_map_util.create_category_index_from_labelmap('mscoco_label_map.pbtxt',
                                                                    use_display_name=True)
cap = cv2.VideoCapture(0)
model = tf.saved_model.load('saved_model')


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
        if dict['detection_classes'][index] == 1 and score > .5:
            label = category_index[dict['detection_classes'][index]]['name']
            ymin, xmin, ymax, xmax = dict['detection_boxes'][index]
            print(box_cords_to_real_location(800, 600, ymin, xmin, ymax, xmax))


def box_cords_to_real_location(img_width, img_height, ymin, xmin, ymax, xmax):
    return [int(img_height * ymin), int(img_width * xmin), int(img_height * ymax), int(img_width * xmax)]


def run_inference(model, cap):
    while cap.isOpened():
        ret, image_np = cap.read()
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
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


run_inference(model, cap)
