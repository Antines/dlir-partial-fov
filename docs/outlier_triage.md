# Outlier triage — frozen v1

The full validation recorded **1,080/1,080 completed cases**, **1,024 improvements** and **56 non-improvements** relative to the scanner header. The median is 1.622 mm, but the p95 TRE is 8.635 mm; catastrophic large-start errors remain.

The table below is reconstructed from the *executed notebook's three-decimal console log* rather than from full-precision `cases.jsonl`. Use the new audit script's CSV for authoritative per-case metrics.

| Case | Header TRE | Refined TRE | Change |
|---|---:|---:|---:|
| `158035 / t32_p30_large` | 9.728 | 25.488 | +15.760 mm |
| `165638 / t48_p70_large_cor` | 10.508 | 26.111 | +15.603 mm |
| `158035 / t48_p70_large` | 10.786 | 26.315 | +15.529 mm |
| `622236 / t48_p70_large_cor` | 10.560 | 25.576 | +15.016 mm |
| `158035 / t48_p30_large` | 10.734 | 25.158 | +14.424 mm |
| `849971 / t48_p30_large` | 9.973 | 21.400 | +11.427 mm |

The large-start subgroup has 15.93% rigid-pose failures versus 2.96% for small starts. Axial cases have 13.70% versus 5.19% for coronal. Slabs 16 mm thick have 14.44% failures, versus 6.94% for both 32 and 48 mm. These figures suggest specific capture and overlap limitations, not a uniformly inaccurate method.

**Review order:** first inspect source EPI and fixed T1 overlays for the five largest regressions; check whether the final similarity score improved despite worse TRE; review loss of brain coverage and physical axis orientation; compare with similarly parameterized cases that succeeded. Do not choose a new optimizer based on the worst cases alone without holding out a separate development subset. Keep the frozen v1 result unchanged.

For all 1,080 cases, `audit_rigid_v1.py` produces `paired_cases.csv`, `subgroups.csv`, `worst_cases.csv`, `regressions.csv`, and `summary.json`. `rigid_stage_qc.py` then generates native-slab-plane T1-to-EPI alignment previews for the top eight worst cases. The QC images are qualitative and do not establish spatial accuracy without the dense truth metrics.
