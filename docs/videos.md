# Videos

Downloaded with yt-dlp into `data/raw/` (not in git). 1080p, video only.

## Milestone 1 (waterline)

| File | Source | Event | Camera |
|------|--------|-------|--------|
| `solo.mp4` | [etyS7HGUXR8](https://youtu.be/etyS7HGUXR8) | Worlds 2025 Singapore, women's solo free (Iris Tió Casas) | underwater, at the water line |

## Milestone 2 (leg / hand detectors), camera above the water

**Train** (label frames from these). All three are the same event and pool: Junior Worlds 2026, Budapest.

| File | Source | Event | Length |
|------|--------|-------|--------|
| `jr2026_w_solo_free_prelim.mp4` | [HdSVdwObn2Y](https://www.youtube.com/live/HdSVdwObn2Y) | Junior Worlds 2026 (Budapest), women's solo free, prelims | 1:57:45 |
| `jr2026_w_solo_tech_final.mp4` | [1XsDoMfpUgQ](https://www.youtube.com/live/1XsDoMfpUgQ) | Junior Worlds 2026, women's solo tech, final | 2:18:35 |
| `jr2026_m_solo_tech_final.mp4` | [vlESM6gJ804](https://www.youtube.com/live/vlESM6gJ804) | Junior Worlds 2026, men's solo tech, final | 1:27:25 |

**Train, batch 2: more pools** (`data/raw/train2/`, 40 frames picked per video, project `M2 legs+arms - train2`).
Added because the batch-1 detector didn't carry over to new pools (docs/problems.md #19).

| File | Source | Event | Pool |
|------|--------|-------|------|
| `fukuoka2023_w_solo_free_final.mp4` | [EX2sh7m25kA](https://youtu.be/EX2sh7m25kA) | Worlds 2023, women's solo free final (Korean broadcast, 22 min) | Fukuoka, indoor |
| `fukuoka2023_tiocasas.mp4` | [cyb_aAoAbh0](https://youtu.be/cyb_aAoAbh0) | Worlds 2023, solo free final (Iris Tió Casas) | Fukuoka, indoor |
| `markham2025_alexandri.mp4` | [a7nBgowwgTg](https://youtu.be/a7nBgowwgTg) | World Cup 2025, solo tech (Vasiliki Alexandri) | Markham, green water |
| `rome2022_cerruti.mp4` | [v56xt7SuqYk](https://youtu.be/v56xt7SuqYk) | European Championships 2022, solo free (Linda Cerruti) | Rome, outdoor |
| `rome2022_planeix.mp4` | [T9h6n-FMsV0](https://youtu.be/T9h6n-FMsV0) | European Championships 2022, solo free (Eve Planeix) | Rome, outdoor |
| `funchal2025_men_solo_team.mp4` | [8_PY-4lLbgE](https://youtu.be/8_PY-4lLbgE) | European Championships 2025, men's solo + team free highlights (team part from 3:05 left out: 8 swimmers per frame) | Funchal, outdoor |

Downloaded but not used: `fukuoka2023_platanioti.mp4` (MOwu3FoDJrg; Platanioti is the Doha test
swimmer), `markham2025_event.mp4` (VLn8r_WC_fc; mostly teams and ceremonies). Note: Platanioti
probably also swims in the 22-min Fukuoka final, a small swimmer overlap with the Doha test video.

**Test** (held out, never used for training). Doha and Paris are different pools from the training videos.

| File | Source | Event | Pool |
|------|--------|-------|------|
| `test/doha2024_worlds_platanioti.mp4` | [W5mwapLlD7I](https://youtu.be/W5mwapLlD7I) | Worlds 2024, solo free final (Evangelia Platanioti) | Doha |
| `test/budapest2024_wc_hoevertsz.mp4` | [fLRuDNBki-w](https://youtu.be/fLRuDNBki-w) | World Cup Super Final 2024, solo free (Kyra Hoevertsz) | Budapest (same city as training, maybe same pool: report its score separately) |
| `test/paris2025_wc_thean.mp4` | [OLUmdnntr7A](https://youtu.be/OLUmdnntr7A) | World Cup 2025, solo free (Rachel Thean) | Paris |
| `test/paris2026_wc_reijnen.mp4` | [EXtTiO2arfU](https://youtu.be/EXtTiO2arfU) | World Cup 2026, solo free (Noortje Reijnen) | Paris |

Note: the test videos are close-ups (fan uploads and a highlights clip); the training videos are
full broadcasts with wider shots. Doha has one underwater shot.
