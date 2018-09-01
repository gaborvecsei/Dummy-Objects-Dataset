from typing import List, Tuple, Union

import numpy as np

from dummy_dataset.shapes import ShapeTypes, Square, Triangle, Circle
from dummy_dataset.utils import non_max_suppression


class DummyShape:
    def __init__(self, shape_type: ShapeTypes, color: Tuple[int, int, int], center_x: int, center_y: int, size: int,
                 label: Union[str, int]):
        self.shape_type = shape_type

        if not shape_type in ShapeTypes:
            raise ValueError("No such shape: {0}".format(self.shape_type))

        self.color = color
        self.x = center_x
        self.y = center_y
        self.size = size
        self.label = label

    @classmethod
    def random_shape_init(cls, parent_image_height: int, parent_image_width: int) -> "DummyShape":
        rnd_shape_type = np.random.choice(list(ShapeTypes))
        rnd_shape_color = tuple([np.random.randint(0, 255) for _ in range(3)])
        margin = 20
        rnd_x = np.random.randint(margin, parent_image_height - margin - 1)
        rnd_y = np.random.randint(margin, parent_image_width - margin - 1)
        rnd_size = np.random.randint(margin, parent_image_height // 4)

        tmp_obj = cls(rnd_shape_type, rnd_shape_color, rnd_x, rnd_y, rnd_size, rnd_shape_type)
        return tmp_obj

    def draw_shape_on_image(self, image: np.ndarray) -> np.ndarray:
        if self.shape_type == ShapeTypes.SQUARE:
            image = Square.draw(image, self.x, self.y, self.size, self.color)
        elif self.shape_type == ShapeTypes.CIRCLE:
            image = Circle.draw(image, self.x, self.y, self.size, self.color)
        elif self.shape_type == ShapeTypes.TRIANGLE:
            Triangle.draw(image, self.x, self.y, self.size, self.color)
        return image


class DummyImage:
    def __init__(self, shapes: List[DummyShape], bg_color: Tuple[int, int, int], height: int, width: int):
        self.shapes = shapes
        self.bg_color = bg_color
        self.height = height
        self.width = width

    def draw_image(self) -> np.ndarray:
        bg_color = np.array(self.bg_color).reshape((1, 1, 3))
        image = np.ones([self.height, self.width, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape in self.shapes:
            image = shape.draw_shape_on_image(image)
        return image

    @classmethod
    def random_image_init(cls, height: int, width: int, min_shapes: int = 1, max_shapes: int = 4) -> "DummyImage":
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

    def get_labels(self) -> Tuple[np.ndarray, np.ndarray]:
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
