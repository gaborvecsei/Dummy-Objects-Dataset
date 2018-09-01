from abc import abstractclassmethod
from enum import Enum
from typing import Tuple

import cv2
import numpy as np


class ShapeTypes(Enum):
    SQUARE = "square"
    CIRCLE = "circle"
    TRIANGLE = "triangle"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class Shape:
    def __init__(self, shape_str: ShapeTypes):
        self.shape_str = shape_str

    @classmethod
    @abstractclassmethod
    def draw(cls, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]) -> np.ndarray:
        pass


class Square(Shape):
    def __init__(self):
        super().__init__(ShapeTypes.SQUARE)

    @classmethod
    def draw(cls, image: np.ndarray, x: int, y: int, size: int,
             color: Tuple[int, int, int]) -> np.ndarray:
        image = cv2.rectangle(image, (x - size, y - size), (x + size, y + size), color,
                              cv2.FILLED)
        return image


class Circle(Shape):
    def __init__(self):
        super().__init__(ShapeTypes.CIRCLE)

    @classmethod
    def draw(cls, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]) -> np.ndarray:
        image = cv2.circle(image, (x, y), size, color, cv2.FILLED)
        return image


class Triangle(Shape):
    def __init__(self):
        super().__init__(ShapeTypes.TRIANGLE)

    @classmethod
    def draw(cls, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]) -> np.ndarray:
        points = np.array([[(x, y - size),
                            (x - size / np.sin(np.radians(60)), y + size),
                            (x + size / np.sin(np.radians(60)), y + size),
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)
        return image
