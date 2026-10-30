# gpu_image_prep concurrency experiments: pre-registered matrix (issue #392)

Written before any run. Results go to `docs/research/v070_measurement_log.md`, section "SP0.7".

Question: why does GPU-route ingest stop scaling past 4 concurrent batches (25 img/s at 4, 20 at 8, GPU 40 to 60 percent)?

Hypotheses and what each predicts:

| Hypothesis | Prediction |
|---|---|
| GIL / host launch overhead inside one `gpu_image_prep` process | one stub process thread near 100 percent CPU; more instances (count 3) raise throughput; fewer worker threads lower the stage inflation |
| Two processes time-slicing one GPU without MPS | GPU util low while instances are busy; count 3 does not help or hurts |
| API side limit | API CPU per image rises with clients while Triton stage times stay flat; throughput tracks API CPU |
| Request queueing (MAX_IN_FLIGHT, LANES) | changing these moves `queue` and the CUDA-call stages |

Common protocol (SP0 runbook): GPU 0 only, under the gpu0 lock, project `optest2`, `ingest_profile.py run` with 3
repetitions, page cache dropped, GPU sampled every 500 ms, load average recorded (above 16 is noisy; noisy repetitions are
repeated when time allows), `--clients 4` and `--clients 8`, set B (500 public photos). Metric: img/s median (min to max), GPU
0 utilisation mean, per-stage table from `ingest_profile.py table`, API CPU-s per image.

Matrix (each row at 4 and 8 clients on set B):

| Id | Change from the committed config (count 2, WORKERS 32, LANES 16, MAX_IN_FLIGHT 96, ENCODE_THREADS 8) |
|---|---|
| C0 | none (control) |
| E-a1 | instance count 1 |
| E-a3 | instance count 3 |
| E-a4 | MAX_IN_FLIGHT 32 |
| E-a5 | MAX_IN_FLIGHT 16 |
| E-c1 | WORKERS 16 |
| E-c2 | WORKERS 8 |
| E-c3 | LANES 8 |
| E-c4 | ENCODE_THREADS 16 |
| E-c5 | Python backend shm-default-byte-size and shm-growth-byte-size 64 MiB |

Then the best configuration and C0 on set P (3,000 sample copies) at 4 clients, and at 8 if it differs, 3 repetitions.

Also: E-d per-thread CPU of the Triton Python stub processes under load (py-spy only if installable and attachable
read-only); E-e API CPU per image at 1, 2, 4, 8, 12 clients with C0.
Dynamic batching of the model is not applicable (`max_batch_size: 0`, decoupled).
Not run: E-b (CUDA MPS on and off). It needs a root MPS daemon; owner decision pending.

Success bar (plan section 5, SP4): set P at least 29 img/s with GPU 0 utilisation above 75 percent at `--clients 4`.
A configuration is a "clear win" when its set B median at 4 or 8 clients beats C0 with non-overlapping min-to-max ranges,
and the set P confirmation holds the direction.
