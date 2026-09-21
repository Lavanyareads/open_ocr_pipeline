import numpy as np
import pytest
from unittest.mock import patch

from app.measurement.barcode_calibration import BarcodeCalibrator
from app.models.schemas import OCRBlock


def _detector_result():
    points = np.array([[[10, 20], [110, 20], [110, 60], [10, 60]]], dtype=np.float32)
    return True, points


def _blocks():
    return [OCRBlock(text="Brand", bbox=[5, 5, 80, 12], image_index=0)]


@patch("app.measurement.barcode_calibration.cv2.imread")
@patch("app.measurement.barcode_calibration.cv2.barcode_BarcodeDetector")
def test_width_calibration(mock_detector_class, mock_imread, tmp_path):
    image_path = tmp_path / "pack.jpg"
    image_path.write_bytes(b"image")
    mock_imread.return_value = object()
    mock_detector_class.return_value.detect.return_value = _detector_result()

    result = BarcodeCalibrator(barcode_width_mm=20).measure(
        [str(image_path)], _blocks()
    )

    assert result["pixels_per_mm"] == 5
    assert result["calibration_dimension"] == "width"
    assert result["text_bboxes"][0]["bbox"] == [5, 5, 80, 12]
    assert result["text_bboxes"][0]["bbox_height_px"] == 12
    assert result["text_bboxes"][0]["estimated_height_mm"] == 2.4


@patch("app.measurement.barcode_calibration.cv2.imread")
@patch("app.measurement.barcode_calibration.cv2.barcode_BarcodeDetector")
def test_height_calibration(mock_detector_class, mock_imread, tmp_path):
    image_path = tmp_path / "pack.jpg"
    image_path.write_bytes(b"image")
    mock_imread.return_value = object()
    mock_detector_class.return_value.detect.return_value = _detector_result()

    result = BarcodeCalibrator(barcode_height_mm=10).measure(
        [str(image_path)], _blocks()
    )

    assert result["pixels_per_mm"] == 4
    assert result["calibration_dimension"] == "height"
    assert result["text_bboxes"][0]["estimated_height_mm"] == 3


@patch("app.measurement.barcode_calibration.cv2.imread")
@patch("app.measurement.barcode_calibration.cv2.barcode_BarcodeDetector")
def test_no_calibration_keeps_pixel_height(mock_detector_class, mock_imread, tmp_path):
    image_path = tmp_path / "pack.jpg"
    image_path.write_bytes(b"image")
    mock_imread.return_value = object()
    mock_detector_class.return_value.detect.return_value = _detector_result()

    result = BarcodeCalibrator().measure([str(image_path)], _blocks())

    assert result["barcode_detected"] is True
    assert result["pixels_per_mm"] is None
    assert result["text_bboxes"][0]["bbox_height_px"] == 12
    assert result["text_bboxes"][0]["estimated_height_mm"] is None


def test_calibration_dimension_must_be_positive():
    with pytest.raises(ValueError):
        BarcodeCalibrator(barcode_width_mm=0)