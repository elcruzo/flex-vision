import json
from pathlib import Path
import numpy as np
import pytest
from cpg.validation import box_iou, check_expected_object, photo_cases

MANIFEST = Path(__file__).resolve().parent/"fixtures/images/manifest.json"


def test_real_fixture_integrity_and_padded_coordinates():
    cases = list(photo_cases(MANIFEST))
    assert len(cases) == 4
    for original, padded in zip(cases[::2], cases[1::2]):
        _, frame, expected, _ = original
        _, canvas, shifted, _ = padded
        np.testing.assert_array_equal(canvas[17:17+frame.shape[0], 29:29+frame.shape[1]], frame)
        np.testing.assert_array_equal(np.array(shifted['box_xyxy'])-expected['box_xyxy'], [29, 17, 29, 17])


def test_checksum_corruption_is_rejected(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data['images'] = [data['images'][0]]
    (tmp_path/data['images'][0]['file']).write_bytes(b'corrupted fixture')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='checksum'):
        list(photo_cases(manifest))


@pytest.mark.parametrize('labels,scores,boxes', [([], [], []), ([17], [.9], [[0, 0, 10, 10]]), ([1], [.1], [[0, 0, 10, 10]]), ([1], [.9], [[30, 30, 40, 40]])])
def test_semantic_or_geometry_failure_cannot_pass(labels, scores, boxes):
    expected = dict(name='person', label=1, min_score=.5, min_iou=.5, box_xyxy=[0, 0, 10, 10])
    with pytest.raises(AssertionError, match='Expected person'):
        check_expected_object(labels, scores, boxes, expected)


def test_iou_and_expected_object_success():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1/3)
    expected = dict(name='cat', label=17, min_score=.5, min_iou=.5, box_xyxy=[0, 0, 10, 10])
    assert check_expected_object([17], [.9], [[0, 0, 10, 10]], expected) == 1
    with pytest.raises(ValueError):
        box_iou([10, 0, 0, 10], [0, 0, 10, 10])


def test_invalid_expectation_cannot_disable_gate():
    expected = dict(name='cat', label=17, min_score=.5, min_iou=float('nan'), box_xyxy=[0, 0, 10, 10])
    with pytest.raises(ValueError, match='thresholds'):
        check_expected_object([], [], [], expected)


def test_fixture_id_cannot_escape_output_directory(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data['images'][0]['id'] = '../../escape'
    manifest = tmp_path/'invalid.json'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Fixture ID'):
        list(photo_cases(manifest))
