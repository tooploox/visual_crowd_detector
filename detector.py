from collections import defaultdict
import time
import operator

import cv2
import numpy as np
import fire
import tensorflow as tf

from danger_zone_detector import (calculate_centers, calculate_cell_centers, find_cell)

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.detection_masks = self.detection_graph.get_tensor_by_name('detection_masks:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        start_time = time.time()
        (masks, boxes, scores, classes, num) = self.sess.run(
            [self.detection_masks, self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        masks_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

            mask = masks[0][i]
            if boxes_list[i][3] - boxes_list[i][1] > 0 and boxes_list[i][2] - boxes_list[i][0]:
                masks_list[i] = cv2.resize(mask, (boxes_list[i][3] - boxes_list[i][1], 
                    boxes_list[i][2] - boxes_list[i][0]), interpolation=cv2.INTER_NEAREST)


        return masks_list, boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def main(model_path='mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
         video_path='scene_1.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = None

    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(video_path)

    total_occupancy = defaultdict(int)
    cell_centers = None

    cnt = 0
    while True:
        r, img = cap.read()
        if img is None:
            break
        cnt += 1
        img = cv2.resize(img, (1280, 720))

        masks, boxes, scores, classes, num = odapi.processFrame(img)

        centers = calculate_centers(boxes)

        if cell_centers is None:
            cell_centers = calculate_cell_centers(img, step_size=200)
            print(cell_centers)

        occupancy = defaultdict(int)
        point_cell = dict()
        for i, (point, class_, score) in enumerate(zip(centers, classes, scores)):
            if not (class_ == 1 and score > threshold):
                continue
            cell_ind = find_cell(cell_centers, point)
            occupancy[cell_ind] += 1
            total_occupancy[cell_ind] += 1
            point_cell[i] = cell_ind

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:

                box = boxes[i]
                mask = masks[i]
                mask = (mask > threshold)

                startX = box[1]
                startY = box[0]
                endX = box[3]
                endY = box[2]
                roi = img[startY:endY, startX:endX]
                roi = roi[mask]

                color = np.array([255.0, 0.0, 0.0])
                person_cell = point_cell[i]
                if occupancy[person_cell] > 2:
                    color = np.array([0.0, 0.0, 255.0])

                blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
                img[startY:endY, startX:endX][mask] = blended

                if cnt > 10:
                    alpha = 0.05
                    overlay = img.copy()
                    sorted_x = sorted(total_occupancy.items(), key=operator.itemgetter(1))
                    top_dangerous = sorted_x[-3:]
                    for dangerous in top_dangerous:
                        center = cell_centers[dangerous[0]]
                        cv2.circle(overlay, (center[0], center[1]), 30, (0, 0, 255), -1)
                        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv2.imshow("preview", img)
        key = cv2.waitKey(200)
        if key & 0xFF == ord('q'):
            break

        if writer is None:
            (h, w) = img.shape[:2]
            writer = \
                cv2.VideoWriter("test_output.mp4", fourcc, 5,
                                (w, h), True)
        writer.write(img)


if __name__ == "__main__":
    fire.Fire(main)

