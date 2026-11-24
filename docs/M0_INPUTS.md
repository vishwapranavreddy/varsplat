# M0 input readiness — 24 November 2026

This inventory distinguishes existing archives, completed extraction checks, and
pending transfers. A metadata directory alone is not a usable RGB-D dataset.
The benchmark runner hashes every associated RGB/depth file and the associated
ground-truth trajectory outside the timed SLAM interval.

| Dataset | Required scenes | Current evidence | Remaining preparation |
|---|---|---|---|
| TUM RGB-D | fr1/desk | Restored archive matches previous cleanup SHA-256; 613 RGB and 595 depth PNG headers checked; upstream associates 592 frames | Running two seed-0 configuration arms; seeds 1/2 remain |
| TUM RGB-D | fr1/desk2, fr1/room, fr2/xyz, fr3/long_office_household | Prior verified checksums and official URLs recovered from the other chat's cleanup records; HTTPS HEAD lengths match | Four resumable archive transfers in progress, 4,817,237,354 bytes total; not yet checksum-verified or extracted in this campaign |
| Replica | room0/1/2, office0/1/2/3/4 | Existing 12,442,855,671-byte archive; prior checksum receipt; current ZIP index has 4,001 files per scene | Scratch extraction and current-run hashing; reference-evaluation assets and headless mesh evaluation preflight |
| ScanNet | 0000_00, 0059_00, 0106_00, 0169_00, 0181_00, 0207_00 | Scene configs present | Dataset access/location remains unresolved; no usable input assets found in inspected locations |
| ScanNet++ | 8b5caf3398, b20a261fdf, 2e74812d00, 281bc17764, fb05e13ad1 | Scene configs present | Dataset access/location remains unresolved; needs DSLR train/test metadata and undistorted RGB/depth |

ScanNet configs 0054_00, 0233_00 and 0465_00 are extra scenes, outside the paper's
six-scene tracking denominator. Do not mix them into that average.

The older experiments have already used Replica and several TUM sequences. They
cannot automatically be called unseen test scenes for a new learned component.
Record a train/development/test split before training, including prior exposure.

## Replica reconstruction reference

The existing archive also contains the eight original scene meshes, but those are
not interchangeable with the culled reference meshes and unseen-region points
required by the upstream evaluator. VarSplat's README points to the
[Co-SLAM evaluation protocol](https://github.com/JingwenWang95/neural_slam_eval),
which supplies separate evaluation data. Its linked archive could not be fetched
through the web reader in this check; availability needs a direct cluster check.
Do not substitute raw meshes or disable reconstruction silently.

The upstream evaluator uses an author-local absolute reference path. A future
scratch mount of the exact reference assets can satisfy that path without editing
SLAM source. Validate the headless Open3D mesh path before committing long Replica
runs. The current TUM campaign does not require those Replica reference assets.

## Staging and storage

`tools/analysis/stage_scene.py` supports the five pinned TUM archives and selective
extraction of one Replica scene. It rejects checksum mismatches, paths outside
allocated scratch, links, duplicate destinations and excessive expanded size.
It checks image signatures/lists and expects 2,000 RGB/depth pairs and poses for
Replica. This is preparation tooling; real multi-scene extraction remains pending.

Replica scene expansions from the current ZIP index range from 1,342,385,731 to
1,816,646,916 bytes. Extract only the scene needed by the current allocation.
No duplicate Replica archive is needed in home.

Whole-home usage before this pair was 48,782,098,432 bytes. The campaign reserved
12 GB for the running pair's returned artifacts and 4.82 GB for the four new TUM
archives, plus setup allowance: below 66 GB combined. This is a reservation, not
a new usage measurement. Check actual usage before the next pair. The 200 GB hard
cap and 50 GB soft/15-day grace rules remain in effect; review staged data by
4 December conservatively. No existing experiment or environment was removed.

Only two batch jobs may be submitted/running across the account. Transfers use
the login node; no training or extraction runs there. All compute, caches and
extracted inputs stay in allocated scratch, with one automatic results copyback.
