import cv2
import numpy as np
import supervision as sv

from vision import ball_utils, cv_utils, detection_utils, robot_utils
from vision.segmentation_config import (
    BALL_ROI_INFER_SIZE_PX,
    BALL_ROI_SIZE_PX,
    CLASS_BALL,
    CLASS_ROBOT,
    CONF_THRESHOLD,
    ROBOT_ROI_INFER_SIZE_PX,
    ROBOT_ROI_SIZE_PX,
    ROI_BALL_PROMPTS,
    ROI_ROBOT_PROMPTS,
)


def infer_ball_roi(
    predictor,
    frame: np.ndarray,
    last_ball_center: tuple[float, float] | None,
) -> tuple[sv.Detections, tuple[int, int, int, int] | None]:
    if last_ball_center is None:
        return sv.Detections.empty(), None

    x1, y1, x2, y2 = detection_utils.roi_bounds(
        frame.shape[1],
        frame.shape[0],
        last_ball_center,
        BALL_ROI_SIZE_PX,
    )
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return sv.Detections.empty(), (x1, y1, x2, y2)

    crop_h, crop_w = crop.shape[:2]
    sam_crop = cv2.resize(
        crop,
        (BALL_ROI_INFER_SIZE_PX, BALL_ROI_INFER_SIZE_PX),
        interpolation=cv2.INTER_CUBIC,
    )

    predictor.set_image(sam_crop)
    roi_results = predictor(text=ROI_BALL_PROMPTS)[0]
    roi_dets = sv.Detections.from_ultralytics(roi_results)
    roi_dets = detection_utils.scale_roi_detections_to_frame(
        roi_dets,
        offset=(x1, y1),
        roi_size=(crop_w, crop_h),
        infer_size=(BALL_ROI_INFER_SIZE_PX, BALL_ROI_INFER_SIZE_PX),
        class_id=CLASS_BALL,
    )

    hsv_dets = ball_utils.hsv_ball_fallback(crop, None, max_candidates=3)
    hsv_dets = detection_utils.offset_roi_detections_to_frame(
        hsv_dets,
        offset=(x1, y1),
        class_id=CLASS_BALL,
    )

    template_dets = ball_utils.template_match_ball(
        crop,
        None,
        search_center=(crop_w / 2.0, crop_h / 2.0),
        search_radius=BALL_ROI_SIZE_PX / 2.0,
    )
    template_dets = detection_utils.offset_roi_detections_to_frame(
        template_dets,
        offset=(x1, y1),
        class_id=CLASS_BALL,
    )

    return detection_utils.combine_detections(
        [roi_dets, hsv_dets, template_dets],
        default_class_id=CLASS_BALL,
    ), (x1, y1, x2, y2)


def infer_missing_robot_rois(
    predictor,
    frame: np.ndarray,
    fresh: sv.Detections,
    robot_cache: sv.Detections | None,
) -> sv.Detections:
    if robot_cache is None or len(robot_cache) == 0:
        return sv.Detections.empty()

    roi_detections = []
    for i in range(len(robot_cache)):
        cached_box = robot_cache.xyxy[i]
        if robot_utils.robot_already_detected(cached_box, fresh):
            continue

        center = cv_utils.center(cached_box)
        x1, y1, x2, y2 = detection_utils.roi_bounds(
            frame.shape[1],
            frame.shape[0],
            center,
            ROBOT_ROI_SIZE_PX,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        sam_crop = cv2.resize(
            crop,
            (ROBOT_ROI_INFER_SIZE_PX, ROBOT_ROI_INFER_SIZE_PX),
            interpolation=cv2.INTER_CUBIC,
        )
        predictor.set_image(sam_crop)
        roi_results = predictor(text=ROI_ROBOT_PROMPTS)[0]
        roi_dets = sv.Detections.from_ultralytics(roi_results)
        roi_dets = detection_utils.scale_roi_detections_to_frame(
            roi_dets,
            offset=(x1, y1),
            roi_size=(crop_w, crop_h),
            infer_size=(ROBOT_ROI_INFER_SIZE_PX, ROBOT_ROI_INFER_SIZE_PX),
            class_id=CLASS_ROBOT,
        )
        roi_dets = roi_dets[roi_dets.confidence > CONF_THRESHOLD] if len(roi_dets) > 0 else roi_dets
        roi_dets = robot_utils.filter_robot_roi_candidates(
            roi_dets,
            cached_box,
            roi_size_px=ROBOT_ROI_SIZE_PX,
        )
        if len(roi_dets) > 0:
            roi_detections.append(roi_dets)

    return detection_utils.combine_detections(
        roi_detections,
        default_class_id=CLASS_ROBOT,
    )
