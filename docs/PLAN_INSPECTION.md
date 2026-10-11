# Static execution-plan inspection

Use the installed development CLI to inspect the current fixed detector graph:

```bash
cpg inspect examples/yolo.yaml --height 1080 --width 1920 --backend cuda --explain
```

The command validates the configuration and shows declared operations, execution groups, and pixel-buffer sizes.
It does not execute preprocessing, import a tensor framework, compile a CUDA kernel, or tune a strategy.
The existing inspect output remains available without --explain.

The detector graph has one group for letterbox, normalization, FP16/FP32 conversion, and NCHW output.
A leading configurable stencil adds a separate specialized pass and an FP32 HWC temporary.
For a 1920×1080 three-channel input, that temporary contains 24,883,200 pixel bytes.
A 1×3×640×640 FP16 output contains 2,457,600 bytes.
These are buffer sizes from dimensions and datatypes, not measured allocation or peak-memory results.
The output may be freshly owned or use the existing optional synchronous caller destination.
Reference-path temporary memory remains unknown because the unfused libraries have different allocation behavior.

Latency, transfers, and peak temporary memory remain null until measured.
Static launch counts describe the selected internal strategy, not a captured driver trace.
Allocator pools, modules, retained inputs, inference allocations, and concurrent frames are excluded from pixel-buffer arithmetic.
The stencil strategy remains experimental and lacks GPU acceptance.

This command advances G4 inspection. It does not implement cpg profile, cpg benchmark, cpg tune, measured plan caching, or general backend selection.
The prepared stencil inference and timing gates remain separate from this static explanation.
