# Independent evidence export

The local coordinator can optionally submit a GPU-side export before the experiment starts.
This export does not require the Mac after submission. GitHub remains optional.
No bucket or persistent volume exists for this project yet. No cloud export passed acceptance.

## Prepared path

Use `scripts/runpod_experiment.py --export-config /private/path/export.json` with the existing experiment arguments.
The private JSON file contains one field: `put_url`.
Supply an HTTPS URL signed for PUT to one unique private object.
Sign with content type `application/gzip`. Keep the URL valid through the entire lease and export allowance.
Do not commit the file or print its contents. Restrict its local permissions to the owner.

The coordinator sends the URL through SSH into an owner-only GPU file.
It sends the stdlib exporter separately. The GPU receives no storage account key.
After the experiment exits, the remote shell saves the execution log and experiment exit code.
It archives the result directory and an internal manifest with file sizes and SHA256 hashes.
It attempts one upload. It removes the remote URL file afterward.
The coordinator waits for that attempt before normal cleanup.
The independent controller still removes the GPU at its deadline, including during a blocked upload.

The exporter rejects symbolic links, empty evidence, and inputs above 512 MiB.
It streams the archive upload. It rejects HTTP redirects and non-success responses.
An upload response does not prove correct retrieval or successful inference.
A missing upload receipt means upload completion is unproven.
The original experiment exit code remains separate from export status.

## Storage setup and acceptance

A signed URL requires an existing private bucket and a signer with permission to write that object.
AWS documents [uploads through signed URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/PresignedUrlUploadObject.html).
A signed upload can replace an existing object. Use a unique experiment/run key.
Choose storage cost and retention before creating a destination. Do not extend the compute lease for export.
Keep model weights and engines outside the result directory unless their redistribution permits upload.

Before another paid inference run:

1. Reconcile the remaining $15 total compute budget.
2. Configure the private destination and its retention rule.
3. Upload harmless evidence through the actual provider.
4. Retrieve the archive independently after the submitting process stops.
5. Verify every manifest entry without extracting untrusted paths or links.
6. Repeat the delayed-consumer experiment only after that export check passes.

The local HTTP receiver check passed archive upload and hash verification after retrieval.
This check validates transport plumbing only. It does not validate a storage provider or CUDA inference.
The preparation does not recover experiment 026's lost evidence.
