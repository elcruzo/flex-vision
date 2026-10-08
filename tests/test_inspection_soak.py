"""Fault checks for the thermal gate. Real acceptance requires the GPU soak."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from inspection_soak import thermal_window


@pytest.mark.parametrize('values, stable', [([60] * 120, True), ([60] * 60 + [62] * 60, False),
                                           ([60, 63] * 60, False)])
def test_thermal_gate(tmp_path, values, stable):
    path = tmp_path / 'telemetry.csv'
    path.write_text(''.join(f'2026/10/08 00:00:00.000, {value}, 1000, 50, 99, 1000\n' for value in values))
    assert thermal_window(path)['stable'] is stable


def test_missing_telemetry_rejects_soak(tmp_path):
    path = tmp_path / 'telemetry.csv'
    path.write_text('')
    with pytest.raises(RuntimeError, match='120'):
        thermal_window(path)
