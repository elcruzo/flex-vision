# Recorded-image smoke fixtures

These two photographs supplement synthetic numerical fixtures.
They do not represent industrial camera data or establish detection accuracy.
The manifest pins original bytes, provenance, broad object boxes, and semantic expectations.

Both files come unchanged from scikit-image revision `e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae` (v0.25.2).
Their SHA256 values match that revision's data registry.

- astronaut.png: NASA photograph, identified as public domain in the upstream data documentation.
- chelsea.png: photograph by Stefan van der Walt, dedicated under CC0 according to the upstream documentation.

See [upstream image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html) for provenance and rights information.
Use these files for technical tests. No endorsement by NASA or the pictured person is implied.

The expected boxes are broad hand-reviewed smoke annotations, not model-generated golden outputs.
The runner also creates padded variants with known offsets to exercise non-square letterbox geometry.
A run must detect the expected class above confidence 0.5 and overlap its source-space box at IoU 0.5 or higher.
It must separately pass tensor, dense model output, and baseline detection comparisons.
