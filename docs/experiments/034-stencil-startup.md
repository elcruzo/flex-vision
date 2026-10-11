# Experiment 034: stencil rental ended before SSH readiness

The full 192-case specialized-stencil gate did not run.
The GPU container did not expose direct SSH within the first five minutes.
The operator ended the exact GPU when fewer than ten minutes remained under the independent 15-minute deadline.
No experiment was submitted, no inference result was exported, and no CUDA acceptance follows from this attempt.

The requested source revision was `2769143` on main.
The controller was `gcz610r8jtkh7o`; the GPU was `tiytub0njqwzr5`.
The controller creation receipt records run `1791683297-298263362463` and the existing source hashes.
GPU termination passed at 2026-10-11 01:53:35 UTC.
The coordinator then confirmed controller and temporary credential cleanup at 01:53:47 UTC.
Fresh API reads returned no pods and no secrets. The conservative accounting timestamp is 01:54:03 UTC.

The quoted GPU rate was $0.89/hour and the controller rate was $0.06/hour.
Replace only this attempt's $0.40 reservation with a $0.207139 lifetime ceiling.
This includes both rates over 346 seconds, another 60 seconds, and $0.10 storage headroom.
The cumulative conservative bound becomes $14.774168 under the original $15 authorization.
These are accounting ceilings, not posted charges. See [accounting](data/budget-after-034.json).
Do not retry under a new $0.40 reservation without further reconciliation or additional authorization.

The coordinator now automatically rejects short-lease startup when ten minutes or less remain.
That check occurs before SSH upload or experiment submission and retains exact GPU cleanup.
The full test protocol and its five-minute export allowance remain unchanged.
The new lifecycle test passes without renting hardware; it does not validate CUDA execution.

The full local suite initially returned 224 passes and one environment failure.
The evidence upload test could not bind its loopback HTTP socket inside the sandbox.
All five evidence-export tests then passed with loopback socket permission.
The new coordinator test also passed. Local checks do not establish full-system acceptance.
