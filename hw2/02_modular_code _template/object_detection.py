"""
Object detection module
"""
from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO
import numpy as np
import numpy.typing as npt
import cv2

# COCO class ids that can block the road: person, bicycle, car, motorcycle, bus, truck.
ROAD_OBSTACLE_CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 5, 7)


@dataclass
class DetectionResult:
    """Obstacles detected in one front-view frame.

    Attributes
    ----------
    boxes_xyxy : (NumDetections x 4) array
        Bounding boxes as [x1, y1, x2, y2] in front-view pixels.
    class_ids : (NumDetections,) array
        COCO class id of every detection.
    confidences : (NumDetections,) array
        Detection confidence in [0, 1].
    masks : (NumDetections x ImageHeight x ImageWidth) array
        Per-detection instance segmentation masks at the front-view resolution.
    class_names : dict[int, str]
        COCO class id -> class name, as reported by the model.
    """
    boxes_xyxy: npt.NDArray[np.float32]
    class_ids: npt.NDArray[np.int64]
    confidences: npt.NDArray[np.float32]
    masks: npt.NDArray[np.bool_]
    class_names: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of detections.

        Returns
        -------
        int
            NumDetections.
        """
        return int(self.class_ids.shape[0])

    def obstacle_mask(self) -> npt.NDArray[np.bool_]:
        """Union of all instance masks.

        Returns
        -------
        (ImageHeight x ImageWidth) array
            True wherever any detected obstacle covers the front-view pixel.
        """
        return self.masks.any(axis=0)


@dataclass
class ObjectDetection:
    """Real-time obstacle segmentation on front-view RGB frames with a pretrained YOLO26 segmentation model.

    Attributes
    ----------
    model_weight_file : str
        Weights file, resolved next to this module. Ultralytics downloads the official weights if it is missing.
    confidence_threshold : float
        Detections below this confidence are dropped.
    obstacle_class_ids : tuple[int, ...]
        COCO class ids kept as obstacles; everything else the model knows is ignored.
    inference_size : int
        Long side, in pixels, the frame is letterboxed to for inference.
    verbose_logging : bool
        Forwarded to Ultralytics' per-call logging.
    box_color : tuple[int, int, int]
        RGB colour of the drawn boxes and labels.
    mask_color : tuple[int, int, int]
        RGB colour blended over segmented pixels.
    mask_alpha : float
        Opacity of the mask overlay in [0, 1].
    box_thickness : int
        Line thickness of the drawn boxes, in pixels.
    """
    model_weight_file: str = "yolo26n-seg.pt"
    confidence_threshold: float = 0.3
    obstacle_class_ids: tuple[int, ...] = ROAD_OBSTACLE_CLASS_IDS
    # The 320x240 camera frames are upscaled 2x; small, distant vehicles are only found at this size, and on a GPU
    # the 640 px pass costs about the same as a 320 px one.
    inference_size: int = 640
    verbose_logging: bool = False
    box_color: tuple[int, int, int] = (0, 255, 0)
    mask_color: tuple[int, int, int] = (255, 0, 255)
    mask_alpha: float = 0.45
    box_thickness: int = 1

    def __post_init__(self) -> None:
        """Load the model once; it is reused for every frame."""
        self._model = YOLO(model=str(Path(__file__).resolve().parent / self.model_weight_file))

    def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult:
        """Segment the road obstacles in a front-view frame.

        Parameters
        ----------
        image : (ImageHeight x ImageWidth x 3) array
            RGB front camera image.

        Returns
        -------
        DetectionResult
            Boxes, classes, confidences and front-view-resolution masks of every obstacle found.
        """
        height, width = image.shape[:2]

        # Ultralytics treats numpy input as OpenCV-style BGR; RGB input swaps red and blue, which mislabels
        # coloured vehicles (e.g. a yellow van as a bus).
        bgr: npt.NDArray[np.uint8] = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # retina_masks upsamples the masks to the input resolution instead of the letterboxed inference grid, so
        # they line up pixel-for-pixel with the frame that front2bev warps.
        result = self._model(
            bgr,
            imgsz=self.inference_size,
            conf=self.confidence_threshold,
            classes=list(self.obstacle_class_ids),
            retina_masks=True,
            verbose=self.verbose_logging,
        )[0].cpu().numpy()

        masks: npt.NDArray[np.bool_] = (
            result.masks.data > 0.5 if result.masks is not None
            else np.zeros((0, height, width), dtype=bool)
        )
        return DetectionResult(
            boxes_xyxy=result.boxes.xyxy.astype(np.float32),
            class_ids=result.boxes.cls.astype(np.int64),
            confidences=result.boxes.conf.astype(np.float32),
            masks=masks,
            class_names=dict(result.names),
        )

    def draw(self, image: npt.NDArray[np.uint8], detections: DetectionResult) -> npt.NDArray[np.uint8]:
        """Overlay masks, boxes and class labels on a copy of the frame.

        Parameters
        ----------
        image : (ImageHeight x ImageWidth x 3) array
            RGB front camera image the detections came from.
        detections : DetectionResult
            Output of detect for this image.

        Returns
        -------
        (ImageHeight x ImageWidth x 3) array
            Annotated RGB image.
        """
        annotated: npt.NDArray[np.uint8] = image.copy()
        covered = detections.obstacle_mask()
        annotated[covered] = (
            (1.0 - self.mask_alpha) * annotated[covered] + self.mask_alpha * np.array(self.mask_color)
        ).astype(np.uint8)

        for box, class_id, confidence in zip(detections.boxes_xyxy, detections.class_ids, detections.confidences):
            # cv2 drawing functions only accept integer points.
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color=self.box_color, thickness=self.box_thickness)
            label = f"{detections.class_names.get(int(class_id), class_id)} {confidence:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.box_color, 1,
                        cv2.LINE_AA)

        return annotated
