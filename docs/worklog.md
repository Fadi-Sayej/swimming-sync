# Work log

| Date | Hours | What I did | Next |
|------|-------|------------|------|
| 2026-10-23 | | Wrote project plan; sent questions to Assaf | Wait for reply |
| 2026-10-25 | | Assaf approved the plan and the product idea (skeleton tool + out-of-sync highlights) | Zoom on Friday |
| 2026-10-29 | | Set up repo, baseline pose script | Zoom with Assaf; get videos and GPU access |
| 2026-10-30 | | Zoom with Assaf. First task: measure how much of the body is out of the water on a good solo YouTube video; run on Mac M4 Pro, college GPU later if needed. Wrote waterline detection, % above water, labeling and evaluation scripts | Run on first video; show results Nov 9 |
| 2026-11-02 | | Ran milestone 1 on Worlds 2025 solo (YouTube, 162 s, 1080p). Fixed waterline (was on blue wall) and swimmer pick (was picking judges). Result: ~22-23% of body above water on average; swimmer found in 74% of frames. Hand-labeled 50 frames: detector error 4.1 px mean (max 9.6), 99.0% above/below agreement. Fixed smoothing across scene cuts. Problems log: docs/problems.md | Prepare Nov 9 results; try a second video |
