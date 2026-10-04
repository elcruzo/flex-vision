import numpy as np
import pytest
from cpg import Pipeline
from cpg.reference import numpy_reference, torch_reference


def pipe(w=7, h=7, encoding="rgb8", dtype="float32"):
    return (Pipeline(input_encoding=encoding).letterbox(w, h)
            .normalize([0, 0, 0], [1, 1, 1], scale=1/255)
            .to(dtype=dtype, layout="NCHW"))


def test_immutable_and_no_silent_execution():
    original = Pipeline()
    _ = original.letterbox(640, 640)
    assert original.operations == ()
    with pytest.raises(TypeError, match="CUDA"):
        pipe()(np.zeros((2, 2, 3), np.uint8))


def test_known_pixels_and_channel_conversion():
    frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    out, g = numpy_reference(pipe(2, 3, "bgr8"), frame)
    np.testing.assert_allclose(out[0, :, 1, :], [[30/255, 60/255], [20/255, 50/255], [10/255, 40/255]])
    np.testing.assert_allclose(out[0, :, 0, :], 114/255)
    assert g.source_box([0, 1, 2, 2]) == (0, 0, 2, 1)


def test_half_pixel_interpolation_known_result():
    frame = np.repeat(np.array([[[0], [100]]], dtype=np.uint8), 3, axis=2)
    out, _ = numpy_reference(pipe(4, 2), frame)
    np.testing.assert_allclose(out[0, 0, 0], np.array([0, 25, 75, 100])/255, atol=1e-7)


@pytest.mark.parametrize("shape,target", [((3, 5, 3), (8, 9)), ((1, 17, 3), (5, 5)), ((17, 1, 3), (5, 5)), ((31, 49, 3), (17, 13))])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_independent_torch_comparison(shape, target, dtype):
    frame = np.random.default_rng(17).integers(0, 256, shape, dtype=np.uint8)
    pipeline = pipe(*target, dtype=dtype)
    expected, geometry = numpy_reference(pipeline, frame)
    actual, actual_geometry = torch_reference(pipeline, frame)
    np.testing.assert_allclose(actual.numpy(), expected, atol=6e-4 if dtype == "float16" else 5e-6, rtol=0)
    assert geometry == actual_geometry
    assert actual.is_contiguous()


def test_strides_and_retained_output():
    frame = np.arange(8*12*3, dtype=np.uint8).reshape(8, 12, 3)[::2, ::2]
    actual, _ = torch_reference(pipe(), frame)
    saved = actual.clone()
    torch_reference(pipe(), np.zeros_like(frame))
    np.testing.assert_array_equal(actual.numpy(), saved.numpy())
    expected, _ = numpy_reference(pipe(), frame)
    np.testing.assert_allclose(actual.numpy(), expected, atol=5e-6, rtol=0)


def test_odd_padding_geometry_and_clipping():
    plan = pipe(6, 6).plan((3, 5, 3))
    assert plan["geometry"]["resized_height"] == 4
    _, g = numpy_reference(pipe(6, 6), np.zeros((3, 5, 3), dtype=np.uint8))
    assert g.source_box([-100, -100, 100, 100]) == (0, 0, 5, 3)
    with pytest.raises(ValueError):
        g.source_box([3, 2, 1, 0])


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_invalid_dimensions(bad):
    with pytest.raises(ValueError):
        Pipeline().letterbox(bad, 10)


def test_reject_invalid_graph_and_input():
    with pytest.raises(ValueError):
        Pipeline().normalize([0]*3, [1]*3, scale=1)
    with pytest.raises(ValueError):
        Pipeline().letterbox(2, 2).normalize([0]*3, [0]*3, scale=1)
    with pytest.raises(ValueError):
        Pipeline().letterbox(2, 2).normalize([0]*3, [1]*3, scale=float("nan"))
    with pytest.raises(ValueError):
        pipe().plan((0, 2, 3))
    with pytest.raises(TypeError):
        numpy_reference(pipe(), np.zeros((2, 2, 3), np.float32))


def test_direct_operations_cannot_bypass_validation():
    from cpg.pipeline import Letterbox, Normalize, Convert
    with pytest.raises(ValueError):
        Letterbox(0, 5, 114)
    with pytest.raises(ValueError):
        Normalize([0, 0, 0], (1, 1, 1), 1)
    with pytest.raises(ValueError):
        Convert("uint8", "NHWC")


def test_explicit_normalization_formula():
    pipeline = (Pipeline().letterbox(1, 1)
                .normalize([1, 2, 3], [2, 4, 5], scale=.5)
                .to(dtype="float32", layout="NCHW"))
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    expected = np.array([2, 2, 2.4], dtype=np.float32)
    actual, _ = numpy_reference(pipeline, frame)
    other, _ = torch_reference(pipeline, frame)
    np.testing.assert_allclose(actual.ravel(), expected)
    np.testing.assert_allclose(other.numpy().ravel(), expected)
