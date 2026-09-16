# Mini-BlobStore

Build a content-addressed object store across five persistent Harbor steps:
immutable blobs, named manifests, multipart uploads, replica repair, and
mark-and-sweep garbage collection. The task is Greenfield and network-isolated.

Each step has independent behavioral tests and cumulative regressions. Dense
analysis dimensions are reported, while `release_pass` requires 100%
correctness. Difficulty remains uncalibrated until target-agent rollouts exist.
