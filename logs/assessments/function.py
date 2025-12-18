import sqlite3
import json
import math
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

SCREEN_WIDTH = 1280.0
SCREEN_HEIGHT = 800.0
TRACKER_NAME_MAP = {
    "pygaze": "A1",
    "pympiigaze": "A2",
    "seeso": "SeeSo",
}


def _mask_t_in_intervals(t_ms: np.ndarray, intervals) -> np.ndarray:
    """Return boolean mask selecting samples where t is inside any [start,end] interval."""
    if not intervals:
        return np.zeros_like(t_ms, dtype=bool)
    m = np.zeros_like(t_ms, dtype=bool)
    for a, b in intervals:
        m |= (t_ms >= float(a)) & (t_ms <= float(b))
    return m


def plot_db_side_over_time(
    db_path,
    max_align_gap_ms=50.0,
    show_gaps=True,
    t_start_s=None,   
    t_end_s=None,     
):

    """
    Plot screen-side (left=0, right=1) over time for all trackers in one DB,
    restricted to the 'images displayed' intervals.

    X: time (s), 0 = first images appearance (within the first interval)
    Y: side (0=left, 1=right)

    - Uses SeeSo timestamps as reference grid and aligns python trackers on nearest timestamps.
    - Only keeps timestamps that are inside images-intervals (test->target or images->target).
    - 'show_gaps=True' inserts NaNs between intervals so lines break (no misleading connections).
    """
    db_path = Path(db_path)

    sess_list = load_trackers_from_db(db_path)
    if not sess_list:
        raise ValueError(f"No tracker data found in {db_path}")

    # Pick reference session (SeeSo)
    ref_list = [s for s in sess_list if str(s["tracker"]) == "SeeSo"]
    if not ref_list:
        raise ValueError(f"SeeSo tracker not found in {db_path}")
    ref = ref_list[0]

    # Define "images are displayed" intervals from EVENTS (protocol-aware)
    events_ref = ref["events"]

    intervals = get_test_images_target_intervals(events_ref)

    if not intervals:
        raise ValueError(f"No test/target (or images/target) intervals found in {db_path}")

    # Time origin = first interval start
    t0 = float(intervals[0][0])

    # Reference gaze timestamps
    ref_gaze = ref["gaze"].copy()
    ref_gaze = ref_gaze.sort_values("timestamp").reset_index(drop=True)

    # Keep only inside intervals
    t_ref = ref_gaze["timestamp"].to_numpy(dtype=float)
    mask_ref = _mask_t_in_intervals(t_ref, intervals)
    ref_gaze = ref_gaze.loc[mask_ref].reset_index(drop=True)

    if ref_gaze.empty:
        raise ValueError(f"SeeSo has no gaze samples inside intervals in {db_path}")

    # Build a "grid" with SeeSo times + side
    screen_mid_x = SCREEN_WIDTH / 2.0
    t_ref = ref_gaze["timestamp"].to_numpy(dtype=float)
    side_ref = (ref_gaze["gx"].to_numpy(dtype=float) >= screen_mid_x).astype(float)  # right=1, left=0

    # Helper: optionally insert NaNs between intervals so the line breaks
    def with_gaps(t_ms, y, intervals):
        if not show_gaps:
            return (t_ms, y)
        # Identify interval index for each timestamp, then break when it changes
        # We'll do a simple pass: assign each t to the first matching interval id.
        ids = np.full(len(t_ms), -1, dtype=int)
        for k, (a, b) in enumerate(intervals):
            m = (t_ms >= float(a)) & (t_ms <= float(b))
            ids[m] = k
        # Insert NaN between successive points belonging to different intervals
        out_t = [t_ms[0]]
        out_y = [y[0]]
        for i in range(1, len(t_ms)):
            if ids[i] != ids[i - 1]:
                out_t.append(np.nan)
                out_y.append(np.nan)
            out_t.append(t_ms[i])
            out_y.append(y[i])
        return (np.array(out_t, dtype=float), np.array(out_y, dtype=float))

    # Align other trackers on SeeSo timeline
    series = {}
    series[ref["tracker"]] = (t_ref, side_ref)

    for s in sess_list:
        name = str(s["tracker"])
        
        if name == ref["tracker"]:
            continue

        other_gaze = s["gaze"].copy()
        other_gaze = other_gaze.sort_values("timestamp").reset_index(drop=True)

        # Align other to ref (within gap)
        aligned = align_gaze_series(ref_gaze, other_gaze, max_gap_ms=max_align_gap_ms)
        if aligned.empty:
            # No aligned samples: keep as NaNs (won't plot)
            continue

        # aligned gives: t_ref, gx_ref, ..., t_other, gx_other, ...
        # We want other side on those aligned rows (already restricted because ref_gaze is restricted)
        t_al = aligned["t_ref"].to_numpy(dtype=float)
        gx_other = aligned["gx_other"].to_numpy(dtype=float)
        side_other = (gx_other >= screen_mid_x).astype(float)

        # For plotting, time starts at first images appearance
        series[name] = (t_al, side_other)

        # ---- Plot ----
        plt.figure(figsize=(12, 4))

        for name, (t_ms, y) in series.items():
            # optionally break lines between intervals
            t_ms2, y2 = with_gaps(t_ms, y, intervals)
            t_s2 = (t_ms2 - t0) / 1000.0

            # ---- ZOOM (apply per-series) ----
            if t_start_s is not None or t_end_s is not None:
                keep = np.ones_like(t_s2, dtype=bool)

                # keep NaNs so line breaks remain visible
                keep |= np.isnan(t_s2)

                if t_start_s is not None:
                    keep &= (t_s2 >= t_start_s) | np.isnan(t_s2)
                if t_end_s is not None:
                    keep &= (t_s2 <= t_end_s) | np.isnan(t_s2)

                t_s2 = t_s2[keep]
                y2 = y2[keep]

            plt.plot(t_s2, y2, linewidth=1.2, label=name)

        plt.yticks([0, 1], ["left (0)", "right (1)"])
        plt.ylim(-0.2, 1.2)
        plt.xlabel("Time since first images (s)")
        plt.ylabel("Screen side")
        plt.title(f"Screen-side over time (images intervals only)\n{db_path.name}")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()



# ---------------------------------------------------------
# Charger tous les trackers d'un .db
# ---------------------------------------------------------

def load_trackers_from_db(db_path: Path):
    """
    Charge un fichier .db et retourne une liste de sessions,
    une par tracker ('seeso', 'pyGaze', 'pyMPIIGaze').

    Chaque session contient :
      - meta
      - events complets
      - images (événements 'images')
      - gaze (événements 'track' pour ce tracker)
    """
    conn = sqlite3.connect(str(db_path))
    meta_df = pd.read_sql_query("SELECT * FROM meta;", conn)
    events_df = pd.read_sql_query("SELECT * FROM events;", conn)
    conn.close()

    if len(meta_df) != 1:
        raise ValueError(f"Meta table in {db_path} has {meta_df.shape[0]} rows, expected 1")

    meta = meta_df.iloc[0].to_dict()

    # ----- Images (left/right ROIs) -----
    img_rows = events_df[events_df["type"] == "images"].copy()
    if not img_rows.empty:
        imgs = img_rows["data"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s else {}
        )
        for side in ["left", "right"]:
            img_rows[f"{side}_left"] = imgs.apply(
                lambda d: d.get(side, {}).get("left", np.nan)
            )
            img_rows[f"{side}_top"] = imgs.apply(
                lambda d: d.get(side, {}).get("top", np.nan)
            )
            img_rows[f"{side}_width"] = imgs.apply(
                lambda d: d.get(side, {}).get("width", np.nan)
            )
            img_rows[f"{side}_height"] = imgs.apply(
                lambda d: d.get(side, {}).get("height", np.nan)
            )

    # ----- Trackers (seeso + pyGaze + pyMPIIGaze) -----
    track_rows = events_df[events_df["type"] == "track"].copy()
    if track_rows.empty:
        return []

    parsed = track_rows["data"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s else None
    )
    track_rows["tracker_name_raw"] = parsed.apply(lambda d: d.get("name") if d else None)

    def _canon_tracker_name(x):
        if x is None:
            return None
        key = str(x).strip().lower()
        return TRACKER_NAME_MAP.get(key, str(x))  # fallback: keep original if unknown

    track_rows["tracker_name"] = track_rows["tracker_name_raw"].apply(_canon_tracker_name)

    track_rows["gaze_json"] = parsed.apply(lambda d: d.get("gaze") if d else None)

    track_rows["gx"] = track_rows["gaze_json"].apply(
        lambda g: g.get("x") if isinstance(g, dict) and "x" in g else np.nan
    )
    track_rows["gy"] = track_rows["gaze_json"].apply(
        lambda g: g.get("y") if isinstance(g, dict) and "y" in g else np.nan
    )
    track_rows["tracker_timestamp"] = track_rows["gaze_json"].apply(
        lambda g: g.get("timestamp") if isinstance(g, dict) and "timestamp" in g else np.nan
    )

    # On garde seulement les samples avec un (x,y) valide
    track_rows = track_rows[~track_rows["gx"].isna()].copy()

    sessions = []
    for tname, sub in track_rows.groupby("tracker_name"):
        if tname in {"A2"}:continue
        sess = {
            "db_path": db_path,
            "meta": meta,
            "identifier": meta.get("identifier"),
            "experiment_type": meta.get("type"),
            "tracker": tname,  
            "events": events_df,
            "images": img_rows,
            "gaze": sub[["sequence", "timestamp", "gx", "gy", "tracker_timestamp"]].copy(),
        }
        sessions.append(sess)

    return sessions


# ---------------------------------------------------------
# Jitter par frame (stabilité temporelle)
# ---------------------------------------------------------

def compute_series_stats(gaze_df: pd.DataFrame):
    """
    Stats de base pour une série de regard :
      - n_samples
      - duration_s
      - fps
      - jitter_mean / jitter_std (distance entre frames successives, en px)
    """
    if gaze_df.empty:
        return {
            "n_samples": 0,
            "duration_s": 0.0,
            "fps": 0.0,
            "jitter_mean": np.nan,
            "jitter_std": np.nan,
        }

    ts = gaze_df["timestamp"].values.astype(float)
    duration_ms = ts.max() - ts.min()
    duration_s = duration_ms / 1000.0 if duration_ms > 0 else 0.0

    n = len(gaze_df)
    fps = n / duration_s if duration_s > 0 else 0.0

    gx = gaze_df["gx"].values.astype(float)
    gy = gaze_df["gy"].values.astype(float)

    if len(gx) >= 2:
        dx = np.diff(gx)
        dy = np.diff(gy)
        step_dist = np.sqrt(dx * dx + dy * dy)
        jitter_mean = float(np.mean(step_dist))
        jitter_std = float(np.std(step_dist))
    else:
        jitter_mean = np.nan
        jitter_std = np.nan

    return {
        "n_samples": int(n),
        "duration_s": float(duration_s),
        "fps": float(fps),
        "jitter_mean": jitter_mean,
        "jitter_std": jitter_std,
    }


# ---------------------------------------------------------
# AOI strict + marges 5%, 10%, 20% pour chaque sample
# ---------------------------------------------------------

def assign_zones_to_gaze(gaze_df: pd.DataFrame,
                         images_df: pd.DataFrame):
    """
    Pour chaque sample de regard (gx, gy, timestamp), associe :
      - zone            : 'left' / 'right' / 'none' (AOI stricte)
      - zone_margin_5   : idem avec marge +5 %
      - zone_margin_10  : idem avec marge +10 %
      - zone_margin_20  : idem avec marge +20 %
      - zone_margin     : alias de zone_margin_20 (compatibilité)
      - dist_to_center  : distance au centre d'AOI (strict) si inside

    Si gaze_df ou images_df est vide, on renvoie des 'none' partout.
    """
    if gaze_df.empty or images_df.empty:
        gaze_df = gaze_df.copy()
        gaze_df["zone"] = "none"
        gaze_df["zone_margin_5"] = "none"
        gaze_df["zone_margin_10"] = "none"
        gaze_df["zone_margin_20"] = "none"
        gaze_df["zone_margin"] = "none"
        gaze_df["dist_to_center"] = np.nan
        return gaze_df

    imgs = images_df.sort_values("timestamp").reset_index(drop=True)
    img_ts = imgs["timestamp"].values

    # Pré-calcul des infos AOI pour chaque événement images
    centers = []
    for _, r in imgs.iterrows():
        row_centers = {}
        for side in ["left", "right"]:
            L = r[f"{side}_left"]
            T = r[f"{side}_top"]
            W = r[f"{side}_width"]
            H = r[f"{side}_height"]
            if not any(pd.isna(v) for v in [L, T, W, H]):
                cx = L + W / 2.0
                cy = T + H / 2.0
                row_centers[side] = {
                    "cx": cx,
                    "cy": cy,
                    "L": L,
                    "T": T,
                    "W": W,
                    "H": H,
                }
        centers.append(row_centers)

    gaze = gaze_df.sort_values("timestamp").copy()
    zones_strict = []
    zones_m5 = []
    zones_m10 = []
    zones_m20 = []
    dists = []

    idx = 0
    for _, gr in gaze.iterrows():
        t = gr["timestamp"]

        # trouver le dernier événement images avant t
        while idx + 1 < len(img_ts) and img_ts[idx + 1] <= t:
            idx += 1

        # cas où le premier event images est après ce gaze
        if img_ts[idx] > t:
            zones_strict.append("none")
            zones_m5.append("none")
            zones_m10.append("none")
            zones_m20.append("none")
            dists.append(np.nan)
            continue

        gx, gy = gr["gx"], gr["gy"]
        row_centers = centers[idx]

        best_zone = "none"
        best_dist = np.nan
        zone_5 = "none"
        zone_10 = "none"
        zone_20 = "none"

        for side, info in row_centers.items():
            cx = info["cx"]
            cy = info["cy"]
            L, T, W, H = info["L"], info["T"], info["W"], info["H"]

            # ROI stricte
            inside_strict = (gx >= L) and (gx <= L + W) and (gy >= T) and (gy <= T + H)

            # ROI étendues
            def inside_ext(ratio: float) -> bool:
                ext_L = L - W * ratio
                ext_T = T - H * ratio
                ext_W = W * (1.0 + 2.0 * ratio)
                ext_H = H * (1.0 + 2.0 * ratio)
                return (
                    gx >= ext_L
                    and gx <= ext_L + ext_W
                    and gy >= ext_T
                    and gy <= ext_T + ext_H
                )

            inside_5 = inside_ext(0.05)
            inside_10 = inside_ext(0.10)
            inside_20 = inside_ext(0.20)

            if inside_strict:
                d = math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                # si plusieurs AOI overlappent, on prend la plus proche
                if best_zone == "none" or d < best_dist:
                    best_zone = side
                    best_dist = d
                    zone_5 = side
                    zone_10 = side
                    zone_20 = side

            else:
                # uniquement si pas déjà fixé par une AOI stricte
                if zone_5 == "none" and inside_5:
                    zone_5 = side
                if zone_10 == "none" and inside_10:
                    zone_10 = side
                if zone_20 == "none" and inside_20:
                    zone_20 = side

        zones_strict.append(best_zone)
        zones_m5.append(zone_5)
        zones_m10.append(zone_10)
        zones_m20.append(zone_20)
        dists.append(best_dist)

    gaze["zone"] = zones_strict
    gaze["zone_margin_5"] = zones_m5
    gaze["zone_margin_10"] = zones_m10
    gaze["zone_margin_20"] = zones_m20
    # compat : zone_margin = +20%
    gaze["zone_margin"] = gaze["zone_margin_20"]
    gaze["dist_to_center"] = dists

    return gaze


# ---------------------------------------------------------
# Intervalles où les deux images sont affichées
# ---------------------------------------------------------

def get_image_display_intervals(images_df: pd.DataFrame):
    """
    Retourne une liste d'intervalles [start, end] (en ms)
    pendant lesquels les deux images (left / right) sont affichées.

    On considère que chaque événement 'images' définit un nouvel écran
    d'images, actif jusqu'au prochain événement 'images'.
    """
    if images_df.empty:
        return []

    intervals = []
    rows = images_df.sort_values("timestamp").reset_index(drop=True)

    for i in range(len(rows)):
        t_start = rows.loc[i, "timestamp"]

        if i + 1 < len(rows):
            t_end = rows.loc[i + 1, "timestamp"]
        else:
            # fallback : 10 secondes après, si rien d'autre
            t_end = t_start + 10_000

        has_left = not any(pd.isna([
            rows.loc[i, "left_left"],
            rows.loc[i, "left_top"],
            rows.loc[i, "left_width"],
            rows.loc[i, "left_height"],
        ]))

        has_right = not any(pd.isna([
            rows.loc[i, "right_left"],
            rows.loc[i, "right_top"],
            rows.loc[i, "right_width"],
            rows.loc[i, "right_height"],
        ]))

        if has_left and has_right:
            intervals.append((t_start, t_end))

    return intervals

def get_test_images_target_intervals(events_df: pd.DataFrame):
    """
    Intervalles [t_first_images_after_test, t_target] pour être strict sur l'affichage réel.
    """
    if events_df.empty:
        return []

    ev = events_df.sort_values("timestamp").reset_index(drop=True)
    types = ev["type"].to_numpy()
    ts = ev["timestamp"].to_numpy(dtype=float)

    intervals = []
    i = 0
    while i < len(ev):
        if types[i] == "test":
            # chercher le premier 'images' après test
            j = i + 1
            while j < len(ev) and types[j] != "images":
                # si on tombe sur target avant images, trial bizarre
                if types[j] == "target":
                    j = None
                    break
                j += 1
            if j is None or j >= len(ev):
                i += 1
                continue
            t_start = ts[j]

            # chercher le prochain 'target' après ce images
            k = j + 1
            while k < len(ev) and types[k] != "target":
                k += 1
            if k < len(ev):
                t_end = ts[k]
                if t_end > t_start:
                    intervals.append((t_start, t_end))
                i = k + 1
            else:
                break
        else:
            i += 1

    return intervals


# ---------------------------------------------------------
# Alignement deux séries (Seeso vs tracker Python)
# ---------------------------------------------------------

def align_gaze_series(ref_gaze: pd.DataFrame,
                      other_gaze: pd.DataFrame,
                      max_gap_ms: float = 50.0) -> pd.DataFrame:
    """
    Aligne deux séries de regard par timestamp le plus proche.
    Retourne un DataFrame avec :
      t_ref, gx_ref, gy_ref,
      t_other, gx_other, gy_other,
      dt_ms
    """
    if ref_gaze.empty or other_gaze.empty:
        return pd.DataFrame(
            columns=["t_ref", "gx_ref", "gy_ref", "t_other", "gx_other", "gy_other", "dt_ms"]
        )

    ref = ref_gaze.sort_values("timestamp").reset_index(drop=True)
    other = other_gaze.sort_values("timestamp").reset_index(drop=True)

    other_ts = other["timestamp"].values
    other_gx = other["gx"].values
    other_gy = other["gy"].values

    aligned_rows = []
    j = 0
    for _, row in ref.iterrows():
        t = row["timestamp"]
        while j + 1 < len(other_ts) and abs(other_ts[j + 1] - t) < abs(other_ts[j] - t):
            j += 1

        dt = other_ts[j] - t
        if abs(dt) <= max_gap_ms:
            aligned_rows.append({
                "t_ref": t,
                "gx_ref": row["gx"],
                "gy_ref": row["gy"],
                "t_other": other_ts[j],
                "gx_other": other_gx[j],
                "gy_other": other_gy[j],
                "dt_ms": dt,
            })

    return pd.DataFrame(aligned_rows)


# ---------------------------------------------------------
# Métriques par paire : erreurs + side agreement + temps en AOI
# ---------------------------------------------------------

def compute_pair_metrics(
    aligned_df: pd.DataFrame,
    zone_strict: pd.Series | None,
    zone_m5: pd.Series | None,
    zone_m10: pd.Series | None,
    zone_m20: pd.Series | None,
    screen_mid_x: float | None,
    image_intervals,
):
    """
    Pour une paire (Seeso, tracker Python) alignée, limitée aux
    timestamps où les images gauche/droite sont affichées :

    - n_aligned
    - err_mean_px / err_rmse_px / err_p95_px : erreurs spatiales vs SeeSo
    - screen_side_agreement_time :
        proportion de paires alignées où SeeSo et le tracker
        sont du même côté de l'écran (gauche/droite)
    - roi_prop_in_aoi_given_side_strict :
        proportion de paires (parmi celles où same_side=True)
        où le tracker est dans l'AOI stricte
    - roi_prop_in_aoi_given_side_5 / 10 / 20 :
        idem pour marges +5%, +10%, +20%

    Pour compatibilité, on définit aussi :
      - roi_time_in_aoi_given_side      = strict
      - roi_time_in_aoi_margin_given_side = +20%
    """
    # rien à aligner ou pas d'intervalles d'images
    if aligned_df.empty or not image_intervals:
        return {
            "n_aligned": 0,
            "err_mean_px": np.nan,
            "err_rmse_px": np.nan,
            "err_p95_px": np.nan,
            "screen_side_agreement_time": np.nan,
            "roi_prop_in_aoi_given_side_strict": np.nan,
            "roi_prop_in_aoi_given_side_5": np.nan,
            "roi_prop_in_aoi_given_side_10": np.nan,
            "roi_prop_in_aoi_given_side_20": np.nan,
            "roi_time_in_aoi_given_side": np.nan,
            "roi_time_in_aoi_margin_given_side": np.nan,
        }

    # --- Filtrer aligned_df pour ne garder que les instants
    #     où les images sont affichées ---
    t = aligned_df["t_ref"].values.astype(float)
    mask = np.zeros(len(t), dtype=bool)                                                                                                                                             
    for start, end in image_intervals:
        mask |= (t >= start) & (t <= end)

    aligned_df = aligned_df[mask].reset_index(drop=True)

    if aligned_df.empty:
        return {
            "n_aligned": 0,
            "err_mean_px": np.nan,
            "err_rmse_px": np.nan,
            "err_p95_px": np.nan,
            "screen_side_agreement_time": np.nan,
            "roi_prop_in_aoi_given_side_strict": np.nan,
            "roi_prop_in_aoi_given_side_5": np.nan,
            "roi_prop_in_aoi_given_side_10": np.nan,
            "roi_prop_in_aoi_given_side_20": np.nan,
            "roi_time_in_aoi_given_side": np.nan,
            "roi_time_in_aoi_margin_given_side": np.nan,
        }

    # --- Filtrer les séries de zones de la même façon ---
    if zone_strict is not None:
        zone_strict = zone_strict[mask].reset_index(drop=True)
    if zone_m5 is not None:
        zone_m5 = zone_m5[mask].reset_index(drop=True)
    if zone_m10 is not None:
        zone_m10 = zone_m10[mask].reset_index(drop=True)
    if zone_m20 is not None:
        zone_m20 = zone_m20[mask].reset_index(drop=True)

    # ----- erreurs spatiales (sur les instants filtrés) -----
    dx = aligned_df["gx_other"].values - aligned_df["gx_ref"].values
    dy = aligned_df["gy_other"].values - aligned_df["gy_ref"].values
    dist = np.sqrt(dx * dx + dy * dy)

    metrics = {
        "n_aligned": int(len(aligned_df)),
        "err_mean_px": float(np.mean(dist)),
        "err_rmse_px": float(math.sqrt(np.mean(dist * dist))),
        "err_p95_px": float(np.percentile(dist, 95)),
        "screen_side_agreement_time": np.nan,
        "roi_prop_in_aoi_given_side_strict": np.nan,
        "roi_prop_in_aoi_given_side_5": np.nan,
        "roi_prop_in_aoi_given_side_10": np.nan,
        "roi_prop_in_aoi_given_side_20": np.nan,
        # compat
        "roi_time_in_aoi_given_side": np.nan,
        "roi_time_in_aoi_margin_given_side": np.nan,
    }

    # ----- Screen-side agreement & ROI (pair-based) -----
    if screen_mid_x is not None:
        gx_ref = aligned_df["gx_ref"].values
        gx_other = aligned_df["gx_other"].values

        side_ref = np.where(gx_ref < screen_mid_x, "left", "right")
        side_other = np.where(gx_other < screen_mid_x, "left", "right")

        same_side = (side_ref == side_other)
        n_pairs = len(same_side)

        if n_pairs > 0:
            # proportion de paires où les deux trackers sont du même côté
            metrics["screen_side_agreement_time"] = float(np.mean(same_side))

            if (
                zone_strict is not None
                and zone_m5 is not None
                and zone_m10 is not None
                and zone_m20 is not None
            ):
                zs = zone_strict.to_numpy()
                z5 = zone_m5.to_numpy()
                z10 = zone_m10.to_numpy()
                z20 = zone_m20.to_numpy()

                mask_same = same_side
                n_same = int(mask_same.sum())

                if n_same > 0:
                    in_strict = (zs != "none") & mask_same
                    in_5 = (z5 != "none") & mask_same
                    in_10 = (z10 != "none") & mask_same
                    in_20 = (z20 != "none") & mask_same

                    p_strict = float(in_strict.sum() / n_same)
                    p5 = float(in_5.sum() / n_same)
                    p10 = float(in_10.sum() / n_same)
                    p20 = float(in_20.sum() / n_same)

                    metrics["roi_prop_in_aoi_given_side_strict"] = p_strict
                    metrics["roi_prop_in_aoi_given_side_5"] = p5
                    metrics["roi_prop_in_aoi_given_side_10"] = p10
                    metrics["roi_prop_in_aoi_given_side_20"] = p20

                    # compat : anciennes colonnes
                    metrics["roi_time_in_aoi_given_side"] = p_strict
                    metrics["roi_time_in_aoi_margin_given_side"] = p20

    return metrics


# ---------------------------------------------------------
# Analyse globale sur un dossier db/
# ---------------------------------------------------------

def analyze_all_dbs(db_root: str, max_align_gap_ms: float = 50.0):
    """
    Analyse tous les fichiers .db dans db_root.

    Retourne :
      - df_sessions : métriques par tracker (jitter, fps, ...)
      - df_pairs    : métriques par paire (Seeso vs tracker Python)
      - sessions    : liste brute des sessions (pour analyses avancées)
    """
    db_root = Path(db_root)
    db_files = sorted(db_root.glob("*.db"))

    sessions = []

    for db in db_files:
        try:
            sess_list = load_trackers_from_db(db)
        except Exception as e:
            print(f"Warning: could not load {db}: {e}")
            continue

        for s in sess_list:
            gaze_zoned = assign_zones_to_gaze(s["gaze"], s["images"])
            s["gaze_zoned"] = gaze_zoned

            stats_single = compute_series_stats(gaze_zoned)
            s["metrics_single"] = stats_single

            sessions.append(s)

    # ---- df_sessions : une ligne par tracker/passation ----
    session_rows = []
    for s in sessions:
        row = {
            "identifier": s["identifier"],
            "experiment_type": s["experiment_type"],
            "tracker": s["tracker"],
            "db_path": str(s["db_path"]),
        }
        row.update(s["metrics_single"])
        session_rows.append(row)

    df_sessions = pd.DataFrame(session_rows)

    # ---- df_pairs : Seeso vs autres trackers ----
    pair_rows = []
    groups = defaultdict(list)
    for s in sessions:
        key = (s["identifier"], s["experiment_type"])
        groups[key].append(s)

    for (identifier, exp_type), sess_list in groups.items():
        # référence SeeSo
        ref_list = [s for s in sess_list if s["tracker"] == "SeeSo"]
        if not ref_list:
            continue
        ref = ref_list[0]

        imgs = ref["images"]
        # milieu de l'écran fixé par la résolution connue
        screen_mid_x = SCREEN_WIDTH / 2.0

        if not imgs.empty:
            events_ref = ref["events"]
            image_intervals = get_test_images_target_intervals(events_ref)
        else:
            image_intervals = []

        for other in sess_list:
            if other is ref:
                continue

            aligned = align_gaze_series(
                ref["gaze_zoned"], other["gaze_zoned"], max_align_gap_ms
            )
            if aligned.empty:
                continue

            # zones côté SeeSo (on ne s'en sert ici que pour t_ref)
            ref_z = ref["gaze_zoned"][["timestamp", "zone"]].rename(
                columns={"timestamp": "t_ref", "zone": "zone_ref"}
            )

            # zones côté tracker Python : strict + marges 5/10/20
            oth_z = other["gaze_zoned"][[
                "timestamp",
                "zone",
                "zone_margin_5",
                "zone_margin_10",
                "zone_margin_20",
            ]].rename(
                columns={
                    "timestamp": "t_other",
                    "zone": "zone_other",
                    "zone_margin_5": "zone_other_m5",
                    "zone_margin_10": "zone_other_m10",
                    "zone_margin_20": "zone_other_m20",
                }
            )

            aligned2 = aligned.merge(ref_z, on="t_ref", how="left").merge(
                oth_z, on="t_other", how="left"
            )

            metrics_pair = compute_pair_metrics(
                aligned2,
                aligned2["zone_other"],
                aligned2["zone_other_m5"],
                aligned2["zone_other_m10"],
                aligned2["zone_other_m20"],
                screen_mid_x,
                image_intervals,
            )

            row = {
                "identifier": identifier,
                "experiment_type": exp_type,
                "ref_tracker": ref["tracker"],
                "other_tracker": other["tracker"],
                "ref_db_path": str(ref["db_path"]),
                "other_db_path": str(other["db_path"]),
            }
            row.update(metrics_pair)
            pair_rows.append(row)

    df_pairs = pd.DataFrame(pair_rows)
    return df_sessions, df_pairs, sessions
