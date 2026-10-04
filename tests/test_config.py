import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from cpg import Pipeline
from cpg.config import ConfigError, load_pipeline
from cpg.reference import numpy_reference

CONFIG = Path(__file__).resolve().parents[1]/"examples/local-detector.yaml"


def test_equivalent_graph_and_pixels():
    python = (Pipeline(input_encoding="bgr8").letterbox(320, 320)
              .normalize([0]*3, [1]*3, scale=1/255).to(dtype="float32", layout="NCHW"))
    yaml = load_pipeline(CONFIG)
    assert python == yaml
    frame = np.random.default_rng(4).integers(0, 256, (7, 13, 3), dtype=np.uint8)
    np.testing.assert_array_equal(numpy_reference(python, frame)[0], numpy_reference(yaml, frame)[0])


@pytest.mark.parametrize("old,new,error", [
    ("encoding: bgr8", "encoding: bgr8\n  encoding: rgb8", "duplicate key"),
    ("schema_version: 1", "schema_version: true", "schema_version"),
    ("width: 320", "width: true", "pipeline\\[0\\].letterbox"),
    ("std: [1, 1, 1]", "std: [0, 1, 1]", "pipeline\\[1\\].normalize"),
    ("scale: 0.00392156862745098", "scale: .nan", "scale"),
    ("scale: 0.00392156862745098", "scle: 1", "unknown fields"),
    ("mean: [0, 0, 0]", "mean: 0", "pipeline\\[1\\].normalize"),
    ("dtype: float32", "dtype: int8", "output"),
    ("letterbox:", "gaussian:", "pipeline\\[0\\]"),
    ("layout: nchw", "layout: NHWC", "output"),
    ("encoding: bgr8", "encoding: grayscale", "input.encoding"),
])
def test_invalid_config_reports_location(tmp_path, old, new, error):
    path = tmp_path/"invalid.yaml"
    path.write_text(CONFIG.read_text().replace(old, new))
    with pytest.raises(ConfigError, match=error):
        load_pipeline(path)


def test_yaml_is_data_not_python(tmp_path):
    path = tmp_path/"unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['echo should-not-run']")
    with pytest.raises(ConfigError, match="constructor"):
        load_pipeline(path)


def test_cli_and_error_exit(tmp_path):
    command = [sys.executable, "-c", "from cpg.cli import main; main()", "inspect", str(CONFIG), "--height", "1080", "--width", "1920"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["output_shape"] == [1, 3, 320, 320]
    bad = subprocess.run(command[:-1]+["0"], text=True, capture_output=True)
    assert bad.returncode == 2
    assert "input width" in bad.stderr
