# P3 and P4: video (outline only)

Status: outline, 2026-11-13. Not a lane spec. Planned in full only after the P2c exit bar
([00](00_photo_library_plan.md) section 2). Research input: [01](01_feature_research.md) sections 3.6 and 6.

## P3: video processing

| Need | Outline | Tools (from 01) |
|---|---|---|
| Ingest | video files enter the library as `media_type = video` photos-with-duration; poster frame and duration on the image doc | FFmpeg with NVDEC; the engine's GPU decode path |
| Playback | H.264/HEVC web proxies, HLS for long clips, generated in the derivatives store | FFmpeg NVENC |
| Keyframes | shot boundaries plus uniform sampling inside shots, capped per minute | TransNetV2 (MIT) |
| Metadata | already read in P2a-0 (one ExifTool reader, embedded tracks); P3 adds nothing to it | |
| Faces and people | detect on keyframes, track inside shots, one face record per track with time ranges; the best-quality frame is the exemplar; exclusion rules apply to clips | SCRFD + ArcFace (as P2), a tracker (to choose) |
| Search | keyframe frame vectors and OCR; results carry clip time ranges | PE, PP-OCRv5 |
| Live and motion photos | pair the still with its clip | |
| Timeline | the `library_videos` rows of P2a-0 become timeline items (images index docs with `media_type = video`); duration badges; the viewer plays the proxy | P2a-1 grid |
| Synced track playback | the video player and the Places map side by side; the marker follows playback time over the stored track (P2a-0 `media_track_points`), and seeking on the map seeks the video | P2a-0 tracks, P2a-1 track layer |

Exit bar to set when planned: face recall on video keyframes, search recall with time ranges, proxy
throughput.

## P4: video memories

| Need | Outline | Tools |
|---|---|---|
| Clip scoring | keyframe quality × people prominence × motion; segment average and peak picking | Google Clips design (01 section 3.6) |
| Clip choice | best 2-4 s segments per chapter; text-query moments when a prompt names them | Moment-DETR code (MIT; train only on our own data) |
| Montage | mixed stills and clips on the P2c timeline (`memory_timeline` gains a clip item); clip trims snap to beats | P2c renderer |
| Reframing | saliency-aware crop paths for portrait output | AutoFlip-style, to evaluate |
| Exclusion | a clip with a blocked identity in any track is not eligible | P2b rules |

Exit bar to set when planned: owner rating of mixed memories, zero excluded identities, render budget.
