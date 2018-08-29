from typing import List, Tuple, Union

import cv2
import numpy as np


class DummyShape:
    SHAPE_TRIANGLE = "triangle"
    SHAPE_SQUARE = "square"
    SHAPE_CIRCLE = "circle"
    _SHAPES_CLASS_MAPPING = {SHAPE_TRIANGLE: 0, SHAPE_CIRCLE: 1, SHAPE_SQUARE: 2}

    def __init__(self, shape_str: str, color: Tuple[int, int, int], x: int, y: int, size: int, label: Union[str, int]):
        self.shape_str = shape_str

        if self.shape_str not in list(self._SHAPES_CLASS_MAPPING.keys()):
            raise ValueError("No such shape: {0}".format(self.shape_str))

        self.color = color
        self.x = x
        self.y = y
        self.size = size
        self.label = label

    @classmethod
    def random_shape_init(cls, parent_image_height: int, parent_image_width: int):
        shape_str = np.random.choice([DummyShape.SHAPE_SQUARE, DummyShape.SHAPE_TRIANGLE,
                                      DummyShape.SHAPE_CIRCLE])
        shape_color = tuple([np.random.randint(0, 255) for _ in range(3)])
        buffer = 20
        y = np.random.randint(buffer, parent_image_height - buffer - 1)
        x = np.random.randint(buffer, parent_image_width - buffer - 1)
        size = np.random.randint(buffer, parent_image_height // 4)

        tmp_obj = cls(shape_str, shape_color, x, y, size, DummyShape._SHAPES_CLASS_MAPPING[shape_str])
        return tmp_obj

    def draw_shape_on_image(self, image):
        if self.shape_str == DummyShape.SHAPE_SQUARE:
            image = cv2.rectangle(image, (self.x - self.size, self.y - self.size),
                                  (self.x + self.size, self.y + self.size),
                                  self.color, -1)
        elif self.shape_str == DummyShape.SHAPE_CIRCLE:
            image = cv2.circle(image, (self.x, self.y), self.size, self.color, -1)
        elif self.shape_str == DummyShape.SHAPE_TRIANGLE:
            points = np.array([[(self.x, self.y - self.size),
                                (self.x - self.size / np.sin(np.radians(60)), self.y + self.size),
                                (self.x + self.size / np.sin(np.radians(60)), self.y + self.size),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, self.color)
        else:
            raise ValueError("No such shape name as: {0}".format(self.shape_str))
        return image


class DummyImage:
    def __init__(self, shapes: List[DummyShape], bg_color: Tuple[int, int, int], height: int, width: int):
        self.shapes = shapes
        self.bg_color = bg_color
        self.height = height
        self.width = width

    def draw_image(self):
        bg_color = np.array(self.bg_color).reshape((1, 1, 3))
        image = np.ones([self.height, self.width, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape in self.shapes:
            image = shape.draw_shape_on_image(image)
        return image

    @classmethod
    def random_image_init(cls, height: int, width: int, min_shapes: int = 1, max_shapes: int = 4):
        shapes = []
        boxes = []

        bg_color = np.array([np.random.randint(0, 255) for _ in range(3)])
        nb_of_shapes = np.random.randint(min_shapes, max_shapes)

        for _ in range(nb_of_shapes):
            shape = DummyShape.random_shape_init(height, width)
            shapes.append(shape)
            boxes.append([shape.y - shape.size, shape.x - shape.size, shape.y + shape.size, shape.x + shape.size])
        keep_ixs = non_max_suppression(np.array(boxes), np.arange(nb_of_shapes), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]

        tmp_obj = cls(shapes, tuple(bg_color), height, width)
        return tmp_obj

    def get_labels(self):
        bboxes = []
        labels = []

        for s in self.shapes:
            x1 = s.x - s.size
            y1 = s.y - s.size
            x2 = s.x + s.size
            y2 = s.y + s.size
            bbox = [x1, y1, x2, y2]

            bboxes.append(bbox)
            labels.append(s.label)

        return np.array(bboxes), np.array(labels)


class DummyObjectsDataset:
    @classmethod
    def get_batch_data_iterator(cls, image_height, image_width, batch_size: int = 1, min_shapes=1, max_shapes=4):
        while True:
            image_batch = []
            bboxes_batch = []
            labels_batch = []

            for i in range(batch_size):
                dummy_image = DummyImage.random_image_init(image_height, image_width, min_shapes, max_shapes)
                image = dummy_image.draw_image()
                bboxes, labels = dummy_image.get_labels()

                image_batch.append(image)
                bboxes_batch.append(bboxes)
                labels_batch.append(labels)
            yield np.array(image_batch), np.array(bboxes_batch), np.array(labels_batch)


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

