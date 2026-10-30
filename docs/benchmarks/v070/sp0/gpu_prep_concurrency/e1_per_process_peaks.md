Per-process card memory peaks (MiB) from the 500 ms `nvidia-smi --query-compute-apps` samples on GPU 0, E1 rows (set P). A stub is one `gpu_image_prep` instance.

| row | tritonserver | stub 1 | stub 2 | sum |
|---|---|---|---|---|
| pc8hr | 8882 | 8488 | 9144 | 26514 |
| pc4hr | 8886 | 8874 | 8880 | 26640 |
| pc8full | 36496 | 6880 | 6880 | 50256 |
| pc4full | 36496 | 7186 | 7472 | 51154 |
