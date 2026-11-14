# Balcony attributes vs test100 hand labels (conditional macro F1)

Links: `eval_astra_compare\balcony_attributes\data\gt_links.jsonl`; facades with links 74, pairs 526 (dropped 0 with unknown crop ids), GT crops linked 526 (0 linked to several Astra crops, rule `largest`).
GT crops with IoU < 0.5 with every hand GT box left out: 12; ours by crop name only.
Ours: `ours_final_v5_20261110.json` (`final` = voted type label); by crop name 487, by unit IoU >= 0.5 0, no ours balcony there 27.

## Astra, all linked GT crops (514 crops)

| head | macro F1 | accuracy | n |
|---|---|---|---|
| Kind | 0.856 | 0.870 | 514 |
| Material (true open_work) | 0.975 | 0.993 | 270 |
| Floor | 0.880 | 0.934 | 212 |
| Structure | 0.846 | 0.846 | 514 |
| **mean of 4** | 0.889 | 0.911 | |

## Ours, same GT crops (487 crops)

| head | macro F1 | accuracy | n |
|---|---|---|---|
| Kind | 0.838 | 0.881 | 487 |
| Material (true open_work) | 0.887 | 0.955 | 221 |
| Floor | 0.920 | 0.970 | 164 |
| Structure | 0.812 | 0.815 | 487 |
| **mean of 4** | 0.864 | 0.905 | |

## Astra, same crops as ours (487 crops)

| head | macro F1 | accuracy | n |
|---|---|---|---|
| Kind | 0.863 | 0.875 | 487 |
| Material (true open_work) | 0.974 | 0.992 | 258 |
| Floor | 0.905 | 0.953 | 193 |
| Structure | 0.848 | 0.848 | 487 |
| **mean of 4** | 0.897 | 0.917 | |

# Confusion: Astra, all linked GT crops

## Kind

| truth \ pred | open_work | surface_panel | solid | recall |
|---|---|---|---|---|
| open_work | 270 | 0 | 0 | 1.000 |
| surface_panel | 53 | 138 | 0 | 0.723 |
| solid | 6 | 8 | 39 | 0.736 |
| precision | 0.821 | 0.945 | 1.000 | |
| F1 | 0.902 | 0.819 | 0.848 | |

## Material (true open_work)

| truth \ pred | metal | masonry | none | recall |
|---|---|---|---|---|
| metal | 247 | 0 | 0 | 1.000 |
| masonry | 2 | 21 | 0 | 0.913 |
| precision | 0.992 | 1.000 |  | |
| F1 | 0.996 | 0.955 |  | |

## Floor

| truth \ pred | rectangle | circle | recall |
|---|---|---|---|
| rectangle | 170 | 3 | 0.983 |
| circle | 11 | 28 | 0.718 |
| precision | 0.939 | 0.903 | |
| F1 | 0.960 | 0.800 | |

## Structure

| truth \ pred | projecting | juliet | recall |
|---|---|---|---|
| projecting | 214 | 49 | 0.814 |
| juliet | 30 | 221 | 0.880 |
| precision | 0.877 | 0.819 | |
| F1 | 0.844 | 0.848 | |

# Confusion: Ours, same GT crops

## Kind

| truth \ pred | open_work | surface_panel | solid | recall |
|---|---|---|---|---|
| open_work | 221 | 23 | 14 | 0.857 |
| surface_panel | 1 | 167 | 12 | 0.928 |
| solid | 6 | 2 | 41 | 0.837 |
| precision | 0.969 | 0.870 | 0.612 | |
| F1 | 0.909 | 0.898 | 0.707 | |

## Material (true open_work)

| truth \ pred | metal | masonry | none | recall |
|---|---|---|---|---|
| metal | 191 | 10 | 0 | 0.950 |
| masonry | 0 | 20 | 0 | 1.000 |
| precision | 1.000 | 0.667 |  | |
| F1 | 0.974 | 0.800 |  | |

## Floor

| truth \ pred | rectangle | circle | recall |
|---|---|---|---|
| rectangle | 144 | 0 | 1.000 |
| circle | 5 | 15 | 0.750 |
| precision | 0.966 | 1.000 | |
| F1 | 0.983 | 0.857 | |

## Structure

| truth \ pred | projecting | juliet | recall |
|---|---|---|---|
| projecting | 166 | 73 | 0.695 |
| juliet | 17 | 231 | 0.931 |
| precision | 0.907 | 0.760 | |
| F1 | 0.787 | 0.837 | |

# Confusion: Astra, same crops as ours

## Kind

| truth \ pred | open_work | surface_panel | solid | recall |
|---|---|---|---|---|
| open_work | 258 | 0 | 0 | 1.000 |
| surface_panel | 49 | 131 | 0 | 0.728 |
| solid | 5 | 7 | 37 | 0.755 |
| precision | 0.827 | 0.949 | 1.000 | |
| F1 | 0.905 | 0.824 | 0.860 | |

## Material (true open_work)

| truth \ pred | metal | masonry | none | recall |
|---|---|---|---|---|
| metal | 236 | 0 | 0 | 1.000 |
| masonry | 2 | 20 | 0 | 0.909 |
| precision | 0.992 | 1.000 |  | |
| F1 | 0.996 | 0.952 |  | |

## Floor

| truth \ pred | rectangle | circle | recall |
|---|---|---|---|
| rectangle | 161 | 3 | 0.982 |
| circle | 6 | 23 | 0.793 |
| precision | 0.964 | 0.885 | |
| F1 | 0.973 | 0.836 | |

## Structure

| truth \ pred | projecting | juliet | recall |
|---|---|---|---|
| projecting | 195 | 44 | 0.816 |
| juliet | 30 | 218 | 0.879 |
| precision | 0.867 | 0.832 | |
| F1 | 0.841 | 0.855 | |
