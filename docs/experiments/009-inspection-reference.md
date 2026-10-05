# Local inspection-to-classifier feedback loop

Result: all three complete inspection reference cases passed on CPU and MPS preprocessing through real CPU classifier inference.
This prepares the second workload. It does not establish CUDA support, industrial accuracy, or the two-workload performance gate.

Both runs used clean revision `55b2aaa2e7a1c4c2a63acf0a1e035279beafd3c2` on the Mac.
The pinned environment uses PyTorch 2.8.0, TorchVision 0.23.0, and NumPy 2.2.6.
Read [the workload contract and reproduction steps](../INSPECTION_REFERENCE.md).

## Evidence

| Preprocessing | Case | Maximum tensor error | Maximum logit error | Cat-group probability |
| --- | --- | ---: | ---: | ---: |
| cpu | chelsea | 0.00001861 | 0.00001717 | 0.991708 |
| cpu | chelsea-odd-roi | 0.00001874 | 0.00002098 | 0.992156 |
| cpu | chelsea-4k-roi | 0.00004581 | 0.00005627 | 0.996707 |
| mps | chelsea | 0.00001861 | 0.00002384 | 0.991708 |
| mps | chelsea-odd-roi | 0.00001861 | 0.00002384 | 0.992156 |
| mps | chelsea-4k-roi | 0.00004607 | 0.00004768 | 0.996707 |

All tensor errors were below the predefined 0.0002 absolute tolerance.
All logits passed the predefined 0.001 absolute and 0.0001 relative tolerances.
The ordered top-five labels matched in every case, and the expected cat-group checks passed.
The top label was Egyptian cat in each case. The assertion allowed the predefined cat class group, not only this observed label.

The negative control requested an absent dog class. The complete runner exited nonzero and saved the expected failed report.
All 43 existing local checks also passed.
MPS preprocessing completed without enabling CPU fallback. Inference and verification downloads remained explicitly on CPU.
No NVIDIA resource was provisioned for this iteration.

## Artifacts

The ignored local bundle is `benchmark-results/iteration-009-inspection/`.
It contains reports, tensors, logits, and a SHA256 manifest.

| Report | SHA256 |
| --- | --- |
| cpu/report.json | 7b697ae32df1b981c6dd8aa2bcc47989d8872864111f7c9027a1d4a64bca15c6 |
| mps/report.json | a45d69b1bd250e9e22162db0e8d50e2740fb5b06299a148bef8ee85adf0e6ae4 |
| negative/report.json | e9f1a50d6a54532fe244e5e3b83c7c50e2149e6adf76ac2c0cb6d808ffe64c87 |

The fixture is a cat photograph on original, odd, and 4K canvases. It is not an industrial inspection dataset.
The public graph grammar and production CUDA operator catalog are unchanged.
Next, establish an equivalent GPU classifier baseline after verifying independent cloud expiration.
Then test the smallest optimized inspection graph through the same consumer.
