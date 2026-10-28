"""
Object detection module
"""
from dataclasses import dataclass, field

from ultralytics import YOLO
import numpy as np
import numpy.typing as npt
import cv2
import cv2.typing as cvt
import ultralytics

@dataclass
class DetectionResult:
    model_results: list[ultralytics.engine.results.Results] = field(default_factory=list)

    def __post_init__(self):
        if self.has_results():
            # Convert all results to numpy
            self.model_results = [r.numpy() for r in self.model_results]

    def has_results(self) -> bool:
        return len(self.model_results) > 0

    def get_boxes_xyxy(self) -> npt.NDArray[np.uint32]:
        """
        Get the bounding boxes in xyxy format

        Returns
        --------
        (NumBoxes x 4) array
            array of boxes in [x1, y1, x2, y2] format
        """
        empty: npt.NDArray[np.uint32] = np.zeros((1, 4), dtype=np.uint32)

        if not self.has_results():
            return empty

        # numpy raises if there isn't at least one array to concat
        boxes: list[npt.NDArray[np.uint32]] = [empty]
        for res in self.model_results:
            boxes.extend([b.xyxy[0].astype(np.uint32) for b in res.boxes])

        return np.vstack(boxes)


@dataclass
class ObjectDetection:
    model_weight_file: str = "yolo26n.pt"
    verbose_logging: bool = False
    box_color: cvt.Scalar = (0, 255, 0)
    box_thickness: int = 4

    def __post_init__(self):
        self._model = YOLO(model=self.model_weight_file)

    def run(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        result = DetectionResult(self._model(image, stream=False, verbose=self.verbose_logging))

        if not result.has_results():
            return image

        boxed = image.copy()

        for bbox_xyxy in result.get_boxes_xyxy():
            cv2.rectangle(
                boxed, 
                bbox_xyxy[:2],  # (x1, y1)
                bbox_xyxy[2:],  # (x2, y2)
                color=self.box_color, 
                thickness=self.box_thickness
            )

        return boxed
