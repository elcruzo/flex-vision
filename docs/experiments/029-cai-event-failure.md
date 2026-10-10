# Experiment 029: CAI harness event compatibility failure

The first CAI GPU run failed before importing any CAI-only input.
CuPy Event has no query() method in the pinned environment.
The harness incorrectly used the Torch event spelling for a CuPy event.
The failed report contains zero CAI checks and successful stream drainage.
Do not infer either numerical failure or interoperability acceptance for the adapter from this attempt.

Clean measured source: `0a64f2afbc8567e9773fe1ff70c74c5f5c88166a`.
Engine SHA-256: `0c9a0518980bf8d37ee2a0d3be79318c556306a807e0d8dfba4c94577cd4318c`.
All six baseline fixture controls ran before this error.
The independent local NumPy/CPU NMS audit passed all 18 baseline candidate/fixture combinations.
These baseline paths use DLPack and do not validate CAI-only input.

Use CuPy Event.done for both pending boundaries.
The [CuPy 14.2 documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Event.html) defines this completion property.
A local regression check rejects completed events and accepts pending events through that property.
The delays, required pending observations, numerical controls, and 144-case matrix remain unchanged.

The runtime also avoids reading a CAI property when DLPack is available.
A dual-protocol object's unused CAI descriptor can raise or perform extra work.
Prefer DLPack without inspecting that descriptor. A local hostile-property test passed.
This correction changes dispatch inspection, not numerical semantics or CAI ordering.
It requires fresh GPU acceptance after the harness correction.

The coordinator exported 35 hash-verified files at 05:18:38 UTC on October 10, 2026.
GPU ejzciarhjrsplv was absent at 05:18:39 UTC.
Controller f2l7aywblb3nn2 and secret i4nf85q8vg cleanup passed at 05:19:01 UTC.
The fresh 05:19:14 UTC inventory returned no Pods or secrets.
See [failure evidence](data/029/cai.json) and [cleanup](data/029/cleanup.json).
Keep the $0.65 reservation until lifetime/billing reconciliation.
The conservative cumulative bound remains $14.356279233504596 within the original $15 authorization.

Next, run a clean corrected revision through all real-inference scenarios.
Retain this failed attempt. Do not replace it with the later result.
CAI transfer capture, broader device/stride contracts, and production acceptance remain pending.
