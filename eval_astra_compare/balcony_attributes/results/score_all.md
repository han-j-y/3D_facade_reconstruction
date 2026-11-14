# Balcony attributes vs test100 hand labels (conditional macro F1)

Links: `eval_astra_compare\balcony_attributes\data\gt_links.jsonl`; facades with links 74, pairs 526 (dropped 0 with unknown crop ids), GT crops linked 526 (0 linked to several Astra crops, rule `largest`).
Ours: `ours_final_v5_20261110.json` (`final` = voted type label); by crop name 487, by unit IoU >= 0.5 8, no ours balcony there 31.

## Astra, all linked GT crops (526 crops)

| head | macro F1 | accuracy | n |
|---|---|---|---|
| Kind | 0.837 | 0.861 | 526 |
| Material (true open_work) | 0.954 | 0.985 | 275 |
| Floor | 0.866 | 0.925 | 214 |
| Structure | 0.846 | 0.846 | 526 |
| **mean of 4** | 0.876 | 0.904 | |

## Ours, same GT crops (495 crops)

| head | macro F1 | accuracy | n |
|---|---|---|---|
| Kind | 0.842 | 0.883 | 495 |
| Material (true open_work) | 0.898 | 0.956 | 226 |
| Floor | 0.920 | 0.970 | 164 |
| Structure | 0.812 | 0.816 | 495 |
| **mean of 4** | 0.868 | 0.906 | |

## Astra, same crops as ours (495 crops)

| head | macro F1 | accuracy | n |
|---|---|---|---|
| Kind | 0.857 | 0.873 | 495 |
| Material (true open_work) | 0.952 | 0.985 | 263 |
| Floor | 0.905 | 0.953 | 193 |
| Structure | 0.846 | 0.846 | 495 |
| **mean of 4** | 0.890 | 0.914 | |

# Confusion: Astra, all linked GT crops

## Kind

| truth \ pred | open_work | surface_panel | solid | recall |
|---|---|---|---|---|
| open_work | 275 | 0 | 0 | 1.000 |
| surface_panel | 53 | 139 | 0 | 0.724 |
| solid | 12 | 8 | 39 | 0.661 |
| precision | 0.809 | 0.946 | 1.000 | |
| F1 | 0.894 | 0.820 | 0.796 | |

## Material (true open_work)

| truth \ pred | metal | masonry | none | recall |
|---|---|---|---|---|
| metal | 249 | 0 | 0 | 1.000 |
| masonry | 4 | 22 | 0 | 0.846 |
| precision | 0.984 | 1.000 |  | |
| F1 | 0.992 | 0.917 |  | |

## Floor

| truth \ pred | rectangle | circle | recall |
|---|---|---|---|
| rectangle | 170 | 3 | 0.983 |
| circle | 13 | 28 | 0.683 |
| precision | 0.929 | 0.903 | |
| F1 | 0.955 | 0.778 | |

## Structure

| truth \ pred | projecting | juliet | recall |
|---|---|---|---|
| projecting | 216 | 49 | 0.815 |
| juliet | 32 | 229 | 0.877 |
| precision | 0.871 | 0.824 | |
| F1 | 0.842 | 0.850 | |

# Confusion: Ours, same GT crops

## Kind

| truth \ pred | open_work | surface_panel | solid | recall |
|---|---|---|---|---|
| open_work | 226 | 23 | 14 | 0.859 |
| surface_panel | 1 | 168 | 12 | 0.928 |
| solid | 6 | 2 | 43 | 0.843 |
| precision | 0.970 | 0.870 | 0.623 | |
| F1 | 0.911 | 0.898 | 0.717 | |

## Material (true open_work)

| truth \ pred | metal | masonry | none | recall |
|---|---|---|---|---|
| metal | 193 | 10 | 0 | 0.951 |
| masonry | 0 | 23 | 0 | 1.000 |
| precision | 1.000 | 0.697 |  | |
| F1 | 0.975 | 0.821 |  | |

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
| juliet | 18 | 238 | 0.930 |
| precision | 0.902 | 0.765 | |
| F1 | 0.785 | 0.840 | |

# Confusion: Astra, same crops as ours

## Kind

| truth \ pred | open_work | surface_panel | solid | recall |
|---|---|---|---|---|
| open_work | 263 | 0 | 0 | 1.000 |
| surface_panel | 49 | 132 | 0 | 0.729 |
| solid | 7 | 7 | 37 | 0.725 |
| precision | 0.824 | 0.950 | 1.000 | |
| F1 | 0.904 | 0.825 | 0.841 | |

## Material (true open_work)

| truth \ pred | metal | masonry | none | recall |
|---|---|---|---|---|
| metal | 238 | 0 | 0 | 1.000 |
| masonry | 4 | 21 | 0 | 0.840 |
| precision | 0.983 | 1.000 |  | |
| F1 | 0.992 | 0.913 |  | |

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
| juliet | 32 | 224 | 0.875 |
| precision | 0.859 | 0.836 | |
| F1 | 0.837 | 0.855 | |
