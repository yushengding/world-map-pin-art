"""
World Map IoU-Greedy v7 - Iterative error-region refinement
  Stage 1: DP pre-simplification
  Stage 2: Global IoU-Greedy to target
  Stage 3: ITERATIVE refinement:
     a) Compute IoU heatmap, find large error regions
     b) Re-insert removed DP points near large errors
     c) Re-run global greedy to trim back to target
     d) Repeat until convergence
"""

import json
import heapq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.path import Path
from shapely.geometry import shape, MultiPolygon, Polygon, LineString
from shapely.ops import unary_union
from scipy import ndimage

# ============================================================
PANEL_W, PANEL_H = 72, 48
MARGIN = 2.0
DW, DH = PANEL_W - 2*MARGIN, PANEL_H - 2*MARGIN
INCH_TO_CM = 2.54
TARGET = 1200
MIN_PERIM = 3.0
MIN_PTS = 3

PPI = 20
RW, RH = int(PANEL_W * PPI), int(PANEL_H * PPI)

# Adaptive refinement schedule: (min_px, rounds, points_per_region)
# Start with large errors, progressively fix smaller ones
REFINE_SCHEDULE = [
    (200, 3, 3),   # large errors first
    (100, 3, 2),
    (50,  3, 2),
    (30,  3, 1),
    (15,  3, 1),
    (8,   3, 1),   # fine-grained
]
POINTS_PER_ERROR_REGION = 3  # default, overridden by schedule

CONTINENT_COLORS = {
    'Eurasia': '#e74c3c', 'Africa': '#f39c12',
    'North America': '#2ecc71', 'South America': '#9b59b6',
    'Oceania': '#1abc9c',
}

CONTINENT_MAP = {}
for codes, cont in [
    (['AFG','ARM','AZE','BHR','BGD','BTN','BRN','KHM','CHN','CYP','GEO','IND',
      'IDN','IRN','IRQ','ISR','JPN','JOR','KAZ','KWT','KGZ','LAO','LBN','MYS',
      'MDV','MNG','MMR','NPL','OMN','PAK','PSE','PHL','QAT','SAU','SGP','KOR',
      'PRK','LKA','SYR','TWN','TJK','THA','TLS','TKM','ARE','UZB','VNM',
      'YEM','IOT','CCK','CXR','HKG','MAC','RUS','TUR'], 'Asia'),
    (['ALB','AND','AUT','BLR','BEL','BIH','BGR','HRV','CZE','DNK','EST','FIN',
      'FRA','DEU','GRC','HUN','ISL','IRL','ITA','XKX','LVA','LIE','LTU','LUX',
      'MLT','MDA','MCO','MNE','NLD','MKD','NOR','POL','PRT','ROU','SMR',
      'SRB','SVK','SVN','ESP','SWE','CHE','UKR','GBR','VAT','FRO','GGY','IMN',
      'JEY','ALA','GIB','SJM','AKR'], 'Europe'),
    (['DZA','AGO','BEN','BWA','BFA','BDI','CPV','CMR','CAF','TCD','COM','COG',
      'COD','CIV','DJI','EGY','GNQ','ERI','SWZ','ETH','GAB','GMB','GHA','GIN',
      'GNB','KEN','LSO','LBR','LBY','MDG','MWI','MLI','MRT','MUS','MAR','MOZ',
      'NAM','NER','NGA','RWA','STP','SEN','SYC','SLE','SOM','ZAF','SSD','SDN',
      'TZA','TGO','TUN','UGA','ZMB','ZWE','SOL','MYT','REU','SHN','ESH','BIR'], 'Africa'),
    (['ATG','BHS','BRB','BLZ','CAN','CRI','CUB','DMA','DOM','SLV','GRD','GTM',
      'HTI','HND','JAM','MEX','NIC','PAN','KNA','LCA','VCT','TTO','USA',
      'AIA','ABW','BMU','VGB','CYM','CUW','GLP','MTQ','MSR','PRI','BES',
      'SXM','MAF','TCA','VIR','GRL','SPM','CLP'], 'North America'),
    (['ARG','BOL','BRA','CHL','COL','ECU','GUY','PRY','PER','SUR','URY','VEN',
      'GUF','FLK','SGS','BVT','BRI'], 'South America'),
    (['AUS','FJI','KIR','MHL','FSM','NRU','NZL','PLW','PNG','WSM','SLB','TON',
      'TUV','VUT','ASM','COK','PYF','GUM','MNP','NCL','NIU','NFK','PCN','TKL',
      'UMI','WLF','ACI','ATC','CSI'], 'Oceania'),
    (['ATA'], 'Antarctica'),
]:
    for c in codes: CONTINENT_MAP[c] = cont

def guess_cont(lon, lat):
    if lat < -60: return 'Antarctica'
    if lat > 60 and lon < -10: return 'North America'
    if lon < -30: return 'North America' if lat > 15 else 'South America'
    if lon < 60: return 'Europe' if lat > 35 else 'Africa'
    if lon < 150: return 'Asia'
    return 'Oceania'

def proj_miller(lon, lat):
    lr = np.radians(np.clip(lat, -85, 85))
    y = np.degrees(1.25 * np.log(np.tan(np.pi/4 + 0.4*lr)))
    return (lon.copy() if isinstance(lon, np.ndarray) else lon), y

# ============================================================
# 1. Load & merge
# ============================================================
print("Step 1: Loading...")
with open('countries.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

country_data = []
continent_geoms = {}
for feat in data['features']:
    p = feat['properties']
    iso3 = p.get('ISO3166-1-Alpha-3', '')
    name = p.get('name', 'Unknown')
    geom = shape(feat['geometry'])
    if not geom.is_valid: geom = geom.buffer(0)
    co = CONTINENT_MAP.get(iso3) or guess_cont(geom.centroid.x, geom.centroid.y)
    if co == 'Antarctica': continue
    dc = 'Eurasia' if co in ('Europe', 'Asia') else co
    country_data.append((name, dc, geom))
    if co not in continent_geoms: continent_geoms[co] = []
    continent_geoms[co].append(geom)

eg = []
for k in ['Europe', 'Asia']:
    if k in continent_geoms: eg.extend(continent_geoms.pop(k))
if eg: continent_geoms['Eurasia'] = eg

cpoly = {}
for co, gs in continent_geoms.items():
    m = unary_union(gs)
    ps = list(m.geoms) if isinstance(m, MultiPolygon) else [m]
    cpoly[co] = ps
    print(f"  {co}: {len(ps)} polygons")

# ============================================================
# 2. Project
# ============================================================
print("Step 2: Projecting...")

def fix_am(c, co):
    cc = c.copy()
    if cc[:, 0].max() - cc[:, 0].min() > 200:
        cc[cc[:, 0] < 0, 0] += 360
    elif cc[:, 0].mean() < -100 and co == 'Eurasia':
        cc[:, 0] += 360
    return cc

def proj_ring(c, co):
    f = fix_am(c, co)
    x, y = proj_miller(f[:, 0], f[:, 1])
    return np.column_stack([x, y])

ap = []
for co, ps in cpoly.items():
    for p in ps:
        ap.append(proj_ring(np.array(p.exterior.coords), co))
pts = np.vstack(ap)
xn, yn = pts.min(0)
xx, yx = pts.max(0)
xr, yr = xx - xn, yx - yn
sc = min(DW/xr, DH/yr)
ox_off = MARGIN + (DW - xr*sc)/2
oy_off = MARGIN + (DH - yr*sc)/2

def to_panel(r):
    o = np.zeros_like(r)
    o[:, 0] = (r[:, 0] - xn)*sc + ox_off
    o[:, 1] = (r[:, 1] - yn)*sc + oy_off
    return o

polygon_data = []
tp = 0
for co, ps in cpoly.items():
    for p in ps:
        e = to_panel(proj_ring(np.array(p.exterior.coords), co))
        ii = [to_panel(proj_ring(np.array(i.coords), co)) for i in p.interiors]
        d = np.diff(e, axis=0)
        pm = np.sqrt((d**2).sum(1)).sum()
        if pm >= MIN_PERIM:
            polygon_data.append({'c': co, 'e': e, 'i': ii, 'p': pm})
            tp += pm

print(f"  {len(polygon_data)} polygons, {sum(len(p['e']) for p in polygon_data)} pts")

# ============================================================
# 3. Rasterize land
# ============================================================
print(f"Step 3: Rasterizing ({RW}x{RH})...")

land_mask = np.zeros((RH, RW), dtype=bool)

def rasterize(ring, mask, val=True):
    rx = ring[:, 0] / PANEL_W * RW
    ry = ring[:, 1] / PANEL_H * RH
    path = Path(np.column_stack([rx, ry]))
    y0 = max(0, int(np.floor(ry.min())))
    y1 = min(RH, int(np.ceil(ry.max())) + 1)
    x0 = max(0, int(np.floor(rx.min())))
    x1 = min(RW, int(np.ceil(rx.max())) + 1)
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    m = path.contains_points(pts).reshape(yy.shape)
    if val: mask[y0:y1, x0:x1] |= m
    else: mask[y0:y1, x0:x1] &= ~m

for pd in polygon_data:
    rasterize(pd['e'], land_mask, True)
    for i in pd['i']:
        rasterize(i, land_mask, False)

print(f"  Land: {land_mask.sum()} px ({100*land_mask.sum()/(RW*RH):.1f}%)")

# ============================================================
# 4. DP pre-simplification
# ============================================================
DP_MULT = 6
S1_TARGET = TARGET * DP_MULT
print(f"\nStep 4: DP to ~{S1_TARGET}...")

def dp_to_n(ring, target_n):
    if len(ring) <= target_n: return ring.copy()
    line = LineString(ring)
    d = np.diff(ring, axis=0)
    sl = np.sqrt((d**2).sum(1))
    mx = sl.sum() / 4
    mn = max(1e-8, sl[sl > 0].min() / 10)
    best = ring.copy()
    bd = len(ring)
    for _ in range(30):
        mid = (mn + mx) / 2
        s = line.simplify(mid, preserve_topology=True)
        n = len(s.coords)
        if abs(n - target_n) < bd:
            bd = abs(n - target_n)
            best = np.array(s.coords)
        if n > target_n: mn = mid
        elif n < target_n: mx = mid
        else: break
    return best

# Store ALL DP points (not just the target budget) for reinsertion later
dp_data = []
dp_total = 0
for pd in polygon_data:
    e = pd['e']
    budget = max(MIN_PTS + 1, int(round(S1_TARGET * pd['p'] / tp)))
    s = dp_to_n(e, budget) if len(e) > budget else e.copy()
    dp_data.append({
        'c': pd['c'], 'dp_pts': s, 'i': pd['i'],
        'p': pd['p'], 'original': e,
    })
    dp_total += len(s)

print(f"  DP: {dp_total} pts")

# ============================================================
# Helper functions
# ============================================================
def ring_sa(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

def tri_sa(a, b, c):
    return 0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

def tri_lf(a, b, c):
    t = np.array([a, b, c])
    rx = t[:, 0] / PANEL_W * RW
    ry = t[:, 1] / PANEL_H * RH
    x0 = max(0, int(np.floor(rx.min())))
    x1 = min(RW, int(np.ceil(rx.max())) + 1)
    y0 = max(0, int(np.floor(ry.min())))
    y1 = min(RH, int(np.ceil(ry.max())) + 1)
    if x1 <= x0 or y1 <= y0: return 0.5
    path = Path(np.column_stack([rx, ry]))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    if len(pts) == 0: return 0.5
    m = path.contains_points(pts).reshape(yy.shape)
    tpx = m.sum()
    if tpx == 0: return 0.5
    return (land_mask[y0:y1, x0:x1] & m).sum() / tpx

def rem_cost(pts, i, pi, ni, ccw):
    sa = tri_sa(pts[pi], pts[i], pts[ni])
    a = abs(sa)
    if a < 1e-10: return 0.0
    lf = tri_lf(pts[pi], pts[i], pts[ni])
    ins = (sa > 0) == ccw
    return a * (lf if ins else 1.0 - lf)

def run_global_greedy(ring_pts_list, target, ring_meta):
    """Run global greedy simplification.
    ring_pts_list: list of np arrays (points for each ring)
    ring_meta: list of dicts with 'ccw', 'continent', 'interiors'
    Returns: list of surviving point arrays
    """
    ring_states = []
    for ri, pts in enumerate(ring_pts_list):
        n = len(pts)
        if n < 3:
            ring_states.append({
                'pts': pts.copy(), 'alive': np.ones(n, dtype=bool),
                'prev': np.zeros(0, dtype=int), 'next': np.zeros(0, dtype=int),
                'ccw': True, 'count': n,
            })
            continue

        # Detect if ring is closed (first == last point)
        is_closed = np.allclose(pts[0], pts[-1], atol=1e-8)

        alive = np.ones(n, dtype=bool)
        pa = np.arange(-1, n-1)
        na = np.arange(1, n+1)

        if is_closed:
            # Closed ring: last point is duplicate, mark dead
            pa[0] = n - 2
            na[n - 2] = 0
            alive[n - 1] = False
            count = n - 1
        else:
            # Open ring (from re-insertion): all points are real
            pa[0] = n - 1
            na[n - 1] = 0
            count = n

        ccw = ring_meta[ri]['ccw']
        ring_states.append({
            'pts': pts.copy(), 'alive': alive,
            'prev': pa, 'next': na,
            'ccw': ccw, 'count': count,
        })

    heap = []
    costs = {}
    ctr = 0

    for ri, rs in enumerate(ring_states):
        if rs['count'] < 3: continue
        for i in range(len(rs['pts'])):
            if not rs['alive'][i]: continue
            c = rem_cost(rs['pts'], i, rs['prev'][i], rs['next'][i], rs['ccw'])
            costs[(ri, i)] = c
            heapq.heappush(heap, (c, ctr, ri, i))
            ctr += 1

    cur = sum(rs['count'] for rs in ring_states)
    to_rem = cur - target

    done = 0
    while done < to_rem and heap:
        c, _, ri, idx = heapq.heappop(heap)
        rs = ring_states[ri]
        if not rs['alive'][idx]: continue
        k = (ri, idx)
        if k in costs and c > costs[k] + 1e-12:
            heapq.heappush(heap, (costs[k], ctr, ri, idx))
            ctr += 1
            continue
        if rs['count'] <= MIN_PTS:
            costs[k] = float('inf')
            continue

        rs['alive'][idx] = False
        rs['count'] -= 1
        done += 1

        pi, ni = rs['prev'][idx], rs['next'][idx]
        rs['next'][pi] = ni
        rs['prev'][ni] = pi

        for j in [pi, ni]:
            if rs['alive'][j]:
                nc = rem_cost(rs['pts'], j, rs['prev'][j], rs['next'][j], rs['ccw'])
                costs[(ri, j)] = nc
                heapq.heappush(heap, (nc, ctr, ri, j))
                ctr += 1

    return [rs['pts'][rs['alive']] for rs in ring_states]

def compute_iou_full(simplified_rings, ring_meta):
    """Compute IoU and return pred_mask."""
    pred = np.zeros((RH, RW), dtype=bool)
    for ri, ext in enumerate(simplified_rings):
        if len(ext) < 3: continue
        cl = np.vstack([ext, ext[0:1]])
        rx = cl[:, 0] / PANEL_W * RW
        ry = cl[:, 1] / PANEL_H * RH
        path = Path(np.column_stack([rx, ry]))
        y0 = max(0, int(np.floor(ry.min())))
        y1 = min(RH, int(np.ceil(ry.max())) + 1)
        x0 = max(0, int(np.floor(rx.min())))
        x1 = min(RW, int(np.ceil(rx.max())) + 1)
        if x1 <= x0 or y1 <= y0: continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        m = path.contains_points(pts).reshape(yy.shape)
        pred[y0:y1, x0:x1] |= m

        for interior in ring_meta[ri].get('interiors', []):
            if len(interior) < 3: continue
            ci = np.vstack([interior, interior[0:1]])
            rxi = ci[:, 0] / PANEL_W * RW
            ryi = ci[:, 1] / PANEL_H * RH
            pi = Path(np.column_stack([rxi, ryi]))
            ym = max(0, int(np.floor(ryi.min())))
            yM = min(RH, int(np.ceil(ryi.max())) + 1)
            xm = max(0, int(np.floor(rxi.min())))
            xM = min(RW, int(np.ceil(rxi.max())) + 1)
            if xM <= xm or yM <= ym: continue
            yy2, xx2 = np.mgrid[ym:yM, xm:xM]
            p2 = np.column_stack([xx2.ravel(), yy2.ravel()])
            m2 = pi.contains_points(p2).reshape(yy2.shape)
            pred[ym:yM, xm:xM] &= ~m2

    inter = (pred & land_mask).sum()
    union = (pred | land_mask).sum()
    iou = inter / union if union > 0 else 0
    prec = inter / pred.sum() if pred.sum() > 0 else 0
    rec = inter / land_mask.sum() if land_mask.sum() > 0 else 0
    return iou, prec, rec, pred

def find_large_error_regions(pred_mask, land_mask, min_size=20):
    """Find connected components of error pixels above min_size.
    Returns list of (centroid_y, centroid_x, size, type) where type is 'FP' or 'FN'."""
    # False positives (red): pred but not land
    fp = pred_mask & ~land_mask
    # False negatives (blue): land but not pred
    fn = ~pred_mask & land_mask

    regions = []

    for error_mask, err_type in [(fp, 'FP'), (fn, 'FN')]:
        labeled, n_labels = ndimage.label(error_mask)
        if n_labels == 0:
            continue
        sizes = ndimage.sum(error_mask, labeled, range(1, n_labels + 1))
        for label_id in range(1, n_labels + 1):
            size = sizes[label_id - 1]
            if size >= min_size:
                # Find centroid in raster coords
                ys, xs = np.where(labeled == label_id)
                cy, cx = ys.mean(), xs.mean()
                # Convert to panel coords
                panel_x = cx / RW * PANEL_W
                panel_y = cy / RH * PANEL_H
                regions.append({
                    'panel_x': panel_x, 'panel_y': panel_y,
                    'size': int(size), 'type': err_type,
                })

    regions.sort(key=lambda r: r['size'], reverse=True)
    return regions

def find_nearest_original_points(panel_x, panel_y, ring_idx, original_ring, n_points=3):
    """Find n_points from original ring nearest to (panel_x, panel_y)."""
    dists = np.sqrt((original_ring[:, 0] - panel_x)**2 +
                    (original_ring[:, 1] - panel_y)**2)
    # Get n_points nearest, but spaced apart
    indices = []
    sorted_idx = np.argsort(dists)
    for idx in sorted_idx:
        if len(indices) >= n_points:
            break
        # Check it's not too close to already selected
        too_close = False
        for existing in indices:
            d = np.sqrt((original_ring[idx, 0] - original_ring[existing, 0])**2 +
                        (original_ring[idx, 1] - original_ring[existing, 1])**2)
            if d < 0.5:  # min 0.5 inch apart
                too_close = True
                break
        if not too_close:
            indices.append(idx)
    return original_ring[indices] if indices else np.array([])

# ============================================================
# 5. Initial greedy
# ============================================================
print(f"\nStep 5: Initial greedy {dp_total} -> {TARGET}...")

ring_meta = []
ring_pts = []
for dd in dp_data:
    pts = dd['dp_pts']
    ccw = ring_sa(pts) > 0
    ring_meta.append({
        'ccw': ccw, 'continent': dd['c'],
        'interiors': dd['i'], 'original': dd['original'],
        'perimeter': dd['p'],
    })
    ring_pts.append(pts)

import os
CHECKPOINT_FILE = 'world_map_checkpoint.npz'

def save_checkpoint(simplified, ring_meta, iou_val):
    """Save current state to checkpoint file."""
    data = {'n_rings': len(simplified), 'iou': iou_val}
    for i, s in enumerate(simplified):
        data[f'ring_{i}'] = s
        data[f'cont_{i}'] = ring_meta[i]['continent']
    np.savez(CHECKPOINT_FILE, **data)
    print(f"  [CHECKPOINT] Saved: {sum(len(s) for s in simplified)} dots, IoU={iou_val:.4f}")

def load_checkpoint(ring_meta):
    """Load checkpoint if compatible. Returns (simplified, iou) or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        d = np.load(CHECKPOINT_FILE, allow_pickle=True)
        n = int(d['n_rings'])
        if n != len(ring_meta):
            print(f"  [CHECKPOINT] Ring count mismatch ({n} vs {len(ring_meta)}), ignoring")
            return None
        simplified = [d[f'ring_{i}'] for i in range(n)]
        prev_iou = float(d['iou'])
        prev_total = sum(len(s) for s in simplified)
        print(f"  [CHECKPOINT] Loaded: {prev_total} dots, IoU={prev_iou:.4f}")
        return simplified, prev_iou
    except Exception as e:
        print(f"  [CHECKPOINT] Load failed: {e}")
        return None

# Try loading checkpoint first
checkpoint = load_checkpoint(ring_meta)
if checkpoint is not None:
    simplified, prev_iou = checkpoint
    prev_total = sum(len(s) for s in simplified)

    if prev_total < TARGET:
        # Need more dots - add from DP pool then let greedy figure it out
        deficit = TARGET - prev_total
        print(f"  Need {deficit} more dots (target={TARGET}, have={prev_total})")
        # Merge checkpoint with DP points, re-run greedy to TARGET
        merged = []
        for ri in range(len(simplified)):
            # Combine existing simplified + DP points for this ring
            existing = simplified[ri]
            dp_pts = dp_data[ri]['dp_pts']
            # Find DP points not already in simplified (distance > 0.3 inch)
            from scipy.spatial import cKDTree as cKDT
            if len(existing) > 0 and len(dp_pts) > 3:
                tree_ex = cKDT(existing)
                dists, _ = tree_ex.query(dp_pts)
                new_pts = dp_pts[dists > 0.3]
                combined = np.vstack([existing, new_pts]) if len(new_pts) > 0 else existing
            else:
                combined = existing
            merged.append(combined)
        merged_total = sum(len(m) for m in merged)
        print(f"  Merged pool: {merged_total} pts, running greedy to {TARGET}...")
        simplified = run_global_greedy(merged, TARGET, ring_meta)
    elif prev_total > TARGET:
        # Need fewer dots - trim
        print(f"  Trimming {prev_total} -> {TARGET}...")
        simplified = run_global_greedy(simplified, TARGET, ring_meta)

    iou, prec, rec, pred_mask = compute_iou_full(simplified, ring_meta)
    total_dots = sum(len(s) for s in simplified)
    print(f"  From checkpoint: {total_dots} dots, IoU={iou:.4f} ({iou*100:.2f}%)")
else:
    # Fresh start: DP -> greedy
    print("  No checkpoint, starting fresh...")
    simplified = run_global_greedy(ring_pts, TARGET, ring_meta)
    iou, prec, rec, pred_mask = compute_iou_full(simplified, ring_meta)
    total_dots = sum(len(s) for s in simplified)
    print(f"  Initial: {total_dots} dots, IoU={iou:.4f} ({iou*100:.2f}%)")

# ============================================================
# 6. Unified Add / Perturb / Delete loop
# ============================================================
from scipy.spatial import cKDTree

OUTER_CYCLES = 30
PERTURB_PER_CYCLE = 5000
PERTURB_SEARCH = 50
ERROR_THRESHOLDS = [150, 60, 25, 10, 4]
POINTS_PER_REGION_SCHED = [3, 2, 1, 1, 1]

def multi_iou_delta(ring_pts, indices, new_positions):
    """Compute IoU change when moving multiple points simultaneously.
    Rejects moves that cause self-intersection.
    """
    n = len(ring_pts)
    if n < 3: return 0

    # Check self-intersection: build new ring and test with Shapely
    from shapely.geometry import LinearRing as LR
    new_ring = ring_pts.copy()
    for pi, pos in zip(indices, new_positions):
        new_ring[pi] = pos
    try:
        lr = LR(new_ring)
        if not lr.is_simple:
            return -999  # reject: would cause self-intersection
    except Exception:
        return -999

    # Bounding box: all old + new positions + their neighbors
    all_pts = []
    for pi in indices:
        all_pts.append(ring_pts[pi])
        all_pts.append(ring_pts[(pi - 1) % n])
        all_pts.append(ring_pts[(pi + 1) % n])
    for p in new_positions:
        all_pts.append(p)
    corners = np.array(all_pts)

    margin = 2
    x0 = max(0, int(np.floor(corners[:, 0].min() / PANEL_W * RW)) - margin)
    x1 = min(RW, int(np.ceil(corners[:, 0].max() / PANEL_W * RW)) + margin)
    y0 = max(0, int(np.floor(corners[:, 1].min() / PANEL_H * RH)) - margin)
    y1 = min(RH, int(np.ceil(corners[:, 1].max() / PANEL_H * RH)) + margin)
    if x1 <= x0 or y1 <= y0: return 0

    yy, xx = np.mgrid[y0:y1, x0:x1]
    test_pts = np.column_stack([xx.ravel(), yy.ravel()])
    if len(test_pts) == 0: return 0

    # Old ring
    old_closed = np.vstack([ring_pts, ring_pts[0:1]])
    old_path = Path(np.column_stack([old_closed[:, 0]/PANEL_W*RW, old_closed[:, 1]/PANEL_H*RH]))
    old_mask = old_path.contains_points(test_pts).reshape(yy.shape)

    # New ring with all points moved
    new_ring = ring_pts.copy()
    for pi, pos in zip(indices, new_positions):
        new_ring[pi] = pos
    new_closed = np.vstack([new_ring, new_ring[0:1]])
    new_path = Path(np.column_stack([new_closed[:, 0]/PANEL_W*RW, new_closed[:, 1]/PANEL_H*RH]))
    new_mask = new_path.contains_points(test_pts).reshape(yy.shape)

    changed = old_mask != new_mask
    if not changed.any(): return 0
    local_land = land_mask[y0:y1, x0:x1]
    return int(((new_mask == local_land) & changed).sum()) - int(((old_mask == local_land) & changed).sum())

def run_mixed_perturbation(simplified, ring_meta, n_iters, rng, pred_mask=None):
    """Mixed 7-strategy perturbation. All strategies in one loop, randomly chosen.
    Strategies:
      0: single-point random slide
      1: 2-3pt coordinated slide (same offset)
      2: 2-3pt normal push (perpendicular to segment)
      3: 2-3pt differential (independent offsets)
      4: error-guided move (toward nearest error region)
      5: edge bisect + remove (split long edge, drop cheapest neighbor)
      6: cross-ring transfer (move point from low-error to high-error ring)
    """
    trees = [cKDTree(ring_meta[ri]['original']) for ri in range(len(simplified))]
    oidx = []
    for ri in range(len(simplified)):
        _, idx = trees[ri].query(simplified[ri])
        oidx.append(idx)

    # Compute per-point error scores for weighted sampling
    # Each point's score = number of error pixels in its neighborhood
    err_points = None
    point_error_scores = []  # flat list of (ring_idx, point_idx, score)
    err_radius_px = 8  # neighborhood radius in raster pixels

    if pred_mask is not None:
        fp = pred_mask & ~land_mask
        fn = ~pred_mask & land_mask
        err_all = fp | fn
        ys, xs = np.where(err_all)
        if len(ys) > 0:
            err_points = np.column_stack([
                xs / RW * PANEL_W,
                ys / RH * PANEL_H
            ])

    # Compute per-point error scores using the error mask
    # Score = count of error pixels near each point
    for ri in range(len(simplified)):
        pts = simplified[ri]
        for pi in range(len(pts)):
            px = int(pts[pi, 0] / PANEL_W * RW)
            py = int(pts[pi, 1] / PANEL_H * RH)
            x0 = max(0, px - err_radius_px)
            x1 = min(RW, px + err_radius_px + 1)
            y0 = max(0, py - err_radius_px)
            y1 = min(RH, py + err_radius_px + 1)
            score = float(err_all[y0:y1, x0:x1].sum()) + 1.0  # +1 so no point has zero weight
            point_error_scores.append((ri, pi, score))

    # Build weighted sampling: higher error = more likely to be selected
    all_scores = np.array([s[2] for s in point_error_scores])
    all_weights = all_scores / all_scores.sum()

    def pick_weighted_point():
        """Pick a (ring_idx, point_idx) weighted by error score."""
        idx = rng.choice(len(point_error_scores), p=all_weights)
        return point_error_scores[idx][0], point_error_scores[idx][1]

    stats = {i: [0, 0] for i in range(7)}  # [accepted, pixels]

    for it in range(n_iters):
        strategy = rng.randint(7)

        if strategy == 0:
            # --- Single point random slide ---
            ri, pi = pick_weighted_point()
            pts = simplified[ri]; n = len(pts)
            if n < 4: continue
            orig = ring_meta[ri]['original']; no = len(orig)
            off = rng.randint(-PERTURB_SEARCH, PERTURB_SEARCH + 1)
            if off == 0: continue
            ci = (oidx[ri][pi] + off) % (no - 1)
            pos = orig[ci]
            if np.sqrt(((pts[pi] - pos)**2).sum()) < 0.05: continue
            d = multi_iou_delta(pts, [pi], [pos])
            if d > 0:
                simplified[ri][pi] = pos.copy(); oidx[ri][pi] = ci
                stats[0][0] += 1; stats[0][1] += d

        elif strategy in (1, 2, 3):
            # --- Multi-point moves ---
            ri, pi = pick_weighted_point()
            pts = simplified[ri]; n = len(pts)
            if n < 5: continue
            gs = 2 if rng.random() < 0.6 else min(3, n - 1)
            indices = [(pi + k) % n for k in range(gs)]
            orig = ring_meta[ri]['original']; no = len(orig)

            new_pos = []; valid = True
            if strategy == 1:
                # Coordinated slide
                off = rng.randint(-PERTURB_SEARCH, PERTURB_SEARCH + 1)
                if off == 0: continue
                for idx in indices:
                    ci = (oidx[ri][idx] + off) % (no - 1)
                    p = orig[ci]
                    if np.sqrt(((pts[idx] - p)**2).sum()) < 0.05: valid = False; break
                    new_pos.append(p)
            elif strategy == 2:
                # Normal push
                p0, p1 = pts[indices[0]], pts[indices[-1]]
                tang = p1 - p0; tl = np.sqrt((tang**2).sum())
                if tl < 1e-6: continue
                normal = np.array([-tang[1], tang[0]]) / tl
                disp = normal * rng.uniform(-0.8, 0.8)
                for idx in indices:
                    cand = pts[idx] + disp
                    _, ni = trees[ri].query(cand)
                    p = orig[ni]
                    if np.sqrt(((pts[idx] - p)**2).sum()) < 0.05: valid = False; break
                    new_pos.append(p)
            else:
                # Differential
                for idx in indices:
                    off = rng.randint(-PERTURB_SEARCH, PERTURB_SEARCH + 1)
                    if off == 0: off = 1
                    ci = (oidx[ri][idx] + off) % (no - 1)
                    p = orig[ci]
                    if np.sqrt(((pts[idx] - p)**2).sum()) < 0.05: valid = False; break
                    new_pos.append(p)

            if not valid or not new_pos: continue
            d = multi_iou_delta(pts, indices, new_pos)
            if d > 0:
                for idx, p in zip(indices, new_pos):
                    simplified[ri][idx] = p.copy()
                    _, ni = trees[ri].query(p); oidx[ri][idx] = ni
                stats[strategy][0] += 1; stats[strategy][1] += d

        elif strategy == 4:
            # --- Error-guided move: pick high-error point, move toward error ---
            if err_points is None or len(err_points) == 0: continue
            ri, pi = pick_weighted_point()
            pts = simplified[ri]; n = len(pts)
            if n < 4: continue
            # Find nearest error pixel as target
            pt = pts[pi]
            dists_to_err = np.sqrt(((err_points - pt)**2).sum(axis=1))
            target = err_points[np.argmin(dists_to_err)]
            # Move this point toward the error, snap to coastline
            direction = target - pts[pi]
            dl = np.sqrt((direction**2).sum())
            if dl < 0.05: continue
            # Move partway toward the error
            frac = rng.uniform(0.3, 1.0)
            candidate = pts[pi] + direction * frac
            orig = ring_meta[ri]['original']
            _, ni = trees[ri].query(candidate)
            pos = orig[ni]
            if np.sqrt(((pts[pi] - pos)**2).sum()) < 0.05: continue
            d = multi_iou_delta(pts, [pi], [pos])
            if d > 0:
                simplified[ri][pi] = pos.copy(); oidx[ri][pi] = ni
                stats[4][0] += 1; stats[4][1] += d

        elif strategy == 5:
            # --- Edge bisect + remove: pick high-error point, bisect its longest adjacent edge ---
            ri, pi = pick_weighted_point()
            pts = simplified[ri]; n = len(pts)
            if n < 6: continue
            # Find longest edge adjacent to selected point
            e1 = np.sqrt(((pts[pi] - pts[(pi+1)%n])**2).sum())
            e2 = np.sqrt(((pts[pi] - pts[(pi-1)%n])**2).sum())
            ei = pi if e1 >= e2 else (pi - 1) % n
            ni_edge = (ei + 1) % n
            # Midpoint of longest edge
            mid = (pts[ei] + pts[ni_edge]) / 2
            orig = ring_meta[ri]['original']
            _, nearest = trees[ri].query(mid)
            new_pt = orig[nearest]
            # Check not too close to existing points
            min_d = np.sqrt(((pts - new_pt)**2).sum(axis=1)).min()
            if min_d < 0.2: continue
            # Find cheapest point to remove (farthest from any error, or smallest triangle)
            # Use triangle area as proxy
            areas = np.zeros(n)
            for i in range(n):
                p0 = pts[(i-1) % n]; p1 = pts[i]; p2 = pts[(i+1) % n]
                areas[i] = abs(0.5 * ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])))
            # Don't remove the points adjacent to insertion
            areas[ei] = float('inf'); areas[ni_edge] = float('inf')
            remove_idx = np.argmin(areas)
            # Build new ring: insert new_pt after ei, remove remove_idx
            new_ring = np.insert(pts, ei + 1, new_pt, axis=0)
            # Adjust remove index (insertion shifts indices after ei)
            adj_remove = remove_idx + 1 if remove_idx > ei else remove_idx
            new_ring = np.delete(new_ring, adj_remove, axis=0)
            # Evaluate: compare old vs new ring IoU in the affected region
            all_affected = np.vstack([pts, new_ring])
            corners = all_affected
            margin = 2
            x0 = max(0, int(np.floor(corners[:,0].min()/PANEL_W*RW)) - margin)
            x1 = min(RW, int(np.ceil(corners[:,0].max()/PANEL_W*RW)) + margin)
            y0 = max(0, int(np.floor(corners[:,1].min()/PANEL_H*RH)) - margin)
            y1 = min(RH, int(np.ceil(corners[:,1].max()/PANEL_H*RH)) + margin)
            if x1 <= x0 or y1 <= y0: continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            tp = np.column_stack([xx.ravel(), yy.ravel()])
            if len(tp) == 0: continue
            old_c = np.vstack([pts, pts[0:1]])
            old_p = Path(np.column_stack([old_c[:,0]/PANEL_W*RW, old_c[:,1]/PANEL_H*RH]))
            old_m = old_p.contains_points(tp).reshape(yy.shape)
            new_c = np.vstack([new_ring, new_ring[0:1]])
            new_p = Path(np.column_stack([new_c[:,0]/PANEL_W*RW, new_c[:,1]/PANEL_H*RH]))
            new_m = new_p.contains_points(tp).reshape(yy.shape)
            changed = old_m != new_m
            if not changed.any(): continue
            ll = land_mask[y0:y1, x0:x1]
            d = int(((new_m == ll) & changed).sum()) - int(((old_m == ll) & changed).sum())
            if d > 0:
                simplified[ri] = new_ring
                _, new_oidx = trees[ri].query(new_ring)
                oidx[ri] = new_oidx
                stats[5][0] += 1; stats[5][1] += d

        elif strategy == 6:
            # --- Cross-ring point transfer ---
            if len(simplified) < 2: continue
            # Find ring with highest error density and lowest error density
            ring_errors = []
            for ri2 in range(len(simplified)):
                pts2 = simplified[ri2]
                if len(pts2) < 4:
                    ring_errors.append(0)
                    continue
                cl = np.vstack([pts2, pts2[0:1]])
                bb_x0 = max(0, int(pts2[:,0].min()/PANEL_W*RW) - 1)
                bb_x1 = min(RW, int(pts2[:,0].max()/PANEL_W*RW) + 2)
                bb_y0 = max(0, int(pts2[:,1].min()/PANEL_H*RH) - 1)
                bb_y1 = min(RH, int(pts2[:,1].max()/PANEL_H*RH) + 2)
                if bb_x1 <= bb_x0 or bb_y1 <= bb_y0:
                    ring_errors.append(0); continue
                local = land_mask[bb_y0:bb_y1, bb_x0:bb_x1]
                # Rough error: just count mismatched pixels in bbox
                ring_errors.append(local.sum() / max(1, (bb_x1-bb_x0)*(bb_y1-bb_y0)))
            # Pick source (low error, many points) and dest (high error)
            re = np.array(ring_errors)
            candidates_src = [i for i in range(len(simplified)) if len(simplified[i]) > 5]
            candidates_dst = [i for i in range(len(simplified)) if len(simplified[i]) >= 4]
            if not candidates_src or not candidates_dst: continue
            src = candidates_src[rng.randint(len(candidates_src))]
            dst = candidates_dst[rng.randint(len(candidates_dst))]
            if src == dst: continue
            # This is complex and rarely improves, skip for efficiency
            continue

    # Build summary
    names = ['single', 'slide', 'normal', 'diff', 'err-guide', 'bisect', 'xfer']
    parts = []
    total_acc = 0; total_px = 0
    for i in range(7):
        a, p = stats[i]
        if a > 0:
            parts.append(f"{names[i]}={a}({p}px)")
        total_acc += a; total_px += p
    summary = ' '.join(parts) if parts else 'none'
    return total_acc, total_px, summary

def run_add_points(simplified, ring_meta, pred_mask, min_px, pts_per_region):
    """Find error regions and insert original coastline points. Returns new ring lists."""
    regions = find_large_error_regions(pred_mask, land_mask, min_px)
    if not regions:
        return [s.copy() for s in simplified], 0

    new_rings = [s.copy() for s in simplified]
    added = 0
    for region in regions[:50]:
        rx, ry = region['panel_x'], region['panel_y']
        best_ri = -1; best_dist = float('inf')
        for ri, pts in enumerate(simplified):
            if len(pts) < 3: continue
            d = np.sqrt((pts[:, 0] - rx)**2 + (pts[:, 1] - ry)**2).min()
            if d < best_dist: best_dist = d; best_ri = ri
        if best_ri < 0 or best_dist > 10: continue
        orig = ring_meta[best_ri]['original']
        new_pts = find_nearest_original_points(rx, ry, best_ri, orig, n_points=pts_per_region)
        if len(new_pts) == 0: continue
        existing = new_rings[best_ri]
        for pt in new_pts:
            dists = np.sqrt(((existing - pt)**2).sum(axis=1))
            if dists.min() < 0.3: continue
            min_edge_dist = float('inf'); insert_after = np.argmin(dists)
            n_ex = len(existing)
            for ei in range(n_ex):
                ej = (ei + 1) % n_ex
                ev = existing[ej] - existing[ei]
                el = np.sqrt((ev**2).sum())
                if el < 1e-10: continue
                t = np.clip(np.dot(pt - existing[ei], ev) / (el**2), 0, 1)
                d = np.sqrt(((pt - (existing[ei] + t*ev))**2).sum())
                if d < min_edge_dist: min_edge_dist = d; insert_after = ei
            existing = np.insert(existing, insert_after + 1, pt, axis=0)
            added += 1
        new_rings[best_ri] = existing
    return new_rings, added

print(f"\nStep 6: Unified Add/Perturb/Delete optimization ({OUTER_CYCLES} cycles)...")

best_iou = iou
best_simplified = [s.copy() for s in simplified]
best_pred = pred_mask.copy()
rng = np.random.RandomState(42)

DELETE_THRESHOLD = TARGET + 50  # only run greedy when exceeding this
DELETE_EVERY = 5  # also force greedy every N cycles

for cycle in range(OUTER_CYCLES):
    th_idx = min(cycle // 3, len(ERROR_THRESHOLDS) - 1)
    min_px = ERROR_THRESHOLDS[th_idx]
    pts_per = POINTS_PER_REGION_SCHED[th_idx]
    is_last = (cycle == OUTER_CYCLES - 1)
    cur_total = sum(len(s) for s in simplified)

    print(f"\n  === Cycle {cycle+1}/{OUTER_CYCLES} (threshold={min_px}px, pts={cur_total}) ===")

    # --- Phase A: ADD points near error regions ---
    new_rings, n_added = run_add_points(simplified, ring_meta, pred_mask, min_px, pts_per)
    if n_added > 0:
        simplified = new_rings
        # Update meta for new rings
        new_meta = []
        for ri in range(len(simplified)):
            p = simplified[ri]
            ccw = ring_sa(np.vstack([p, p[0:1]])) > 0 if len(p) >= 3 else ring_meta[ri]['ccw']
            new_meta.append({
                'ccw': ccw, 'continent': ring_meta[ri]['continent'],
                'interiors': ring_meta[ri]['interiors'],
                'original': ring_meta[ri]['original'],
                'perimeter': ring_meta[ri]['perimeter'],
            })
        ring_meta = new_meta
        cur_total = sum(len(s) for s in simplified)
        print(f"  [ADD] +{n_added} -> {cur_total} pts")
    else:
        print(f"  [ADD] none at {min_px}px")

    # --- Phase B: DELETE only when needed ---
    need_delete = is_last or cur_total > DELETE_THRESHOLD or (cycle % DELETE_EVERY == DELETE_EVERY - 1)
    if need_delete and cur_total > TARGET:
        simplified = run_global_greedy(simplified, TARGET, ring_meta)
        cur_total = sum(len(s) for s in simplified)
        iou, _, _, pred_mask = compute_iou_full(simplified, ring_meta)
        print(f"  [DEL] -> {cur_total} dots, IoU={iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            best_simplified = [s.copy() for s in simplified]
            best_pred = pred_mask.copy()
            print(f"  ** New best: {best_iou:.4f} **")

    # --- Phase C: Mixed 7-strategy perturbation ---
    acc, px, summary = run_mixed_perturbation(
        simplified, ring_meta, PERTURB_PER_CYCLE, rng, pred_mask)
    print(f"  [PERTURB] {summary}")

    iou, _, _, pred_mask = compute_iou_full(simplified, ring_meta)
    cur_total = sum(len(s) for s in simplified)
    print(f"  Result: {cur_total} dots, IoU={iou:.4f} ({iou*100:.2f}%)")

    if iou > best_iou:
        best_iou = iou
        best_simplified = [s.copy() for s in simplified]
        best_pred = pred_mask.copy()
        save_checkpoint(simplified, ring_meta, iou)
        print(f"  ** New best: {best_iou:.4f} **")
    elif cur_total <= TARGET:
        simplified = [s.copy() for s in best_simplified]
        pred_mask = best_pred.copy()
        iou = best_iou
        print(f"  (Reverted to best: {best_iou:.4f})")

# Final: ensure exactly TARGET dots
cur_total = sum(len(s) for s in simplified)
if cur_total > TARGET:
    print(f"\n  Final trim: {cur_total} -> {TARGET}...")
    simplified = run_global_greedy(simplified, TARGET, ring_meta)
    iou, _, _, pred_mask = compute_iou_full(simplified, ring_meta)
    if iou > best_iou:
        best_iou = iou
        best_simplified = [s.copy() for s in simplified]
        best_pred = pred_mask.copy()

simplified = best_simplified
pred_mask = best_pred
iou = best_iou

# ============================================================
# 6c. Fix self-intersections
# ============================================================
from shapely.geometry import LinearRing as LR
print("\nStep 6c: Fixing self-intersections...")
n_fixed = 0
for ri in range(len(simplified)):
    pts = simplified[ri]
    if len(pts) < 4: continue
    lr = LR(pts)
    if not lr.is_simple:
        print(f"  Ring {ri} ({ring_meta[ri]['continent']}): self-intersecting! ({len(pts)} pts)")
        # Fix: make_valid using Shapely buffer(0)
        from shapely.geometry import Polygon as ShapelyPoly
        poly = ShapelyPoly(pts)
        fixed = poly.buffer(0)
        if fixed.is_empty: continue
        # Extract the largest polygon from the result
        if hasattr(fixed, 'geoms'):
            areas = [g.area for g in fixed.geoms]
            fixed = fixed.geoms[np.argmax(areas)]
        new_pts = np.array(fixed.exterior.coords)[:-1]  # remove closing duplicate
        if len(new_pts) >= 3:
            simplified[ri] = new_pts
            n_fixed += 1
            print(f"    Fixed: {len(pts)} -> {len(new_pts)} pts")

if n_fixed > 0:
    # Recount and possibly trim
    cur_total = sum(len(s) for s in simplified)
    if cur_total > TARGET:
        simplified = run_global_greedy(simplified, TARGET, ring_meta)
    iou, _, _, pred_mask = compute_iou_full(simplified, ring_meta)
    print(f"  After fix: IoU={iou:.4f} ({iou*100:.2f}%)")
else:
    print("  No self-intersections found.")

# ============================================================
# 6d. Focused optimization on user-marked problem areas
# ============================================================
# Areas identified from user's screenshot annotation (panel coordinates)
FOCUS_REGIONS = [
    ('Alaska/Aleutians', 3, 33, 5, 3),       # x_center, y_center, radius (inches)
    ('Canadian Arctic', 15, 39, 5, 3),
    ('Greenland W', 22, 40, 3, 2),
    ('Caribbean', 18, 28, 3, 2),
    ('Chile S', 20, 8, 3, 3),
    ('Scandinavia', 34, 40, 3, 2),
    ('Japan/Korea', 57, 32, 3, 2),
    ('Kamchatka', 62, 36, 3, 2),
    ('Madagascar', 45, 18, 2, 2),
]

print(f"\nStep 6d: Focused optimization on {len(FOCUS_REGIONS)} marked regions...")

# For each focus region, do intensive perturbation on nearby points
trees_f = [cKDTree(ring_meta[ri]['original']) for ri in range(len(simplified))]

for region_name, cx, cy, rx, ry in FOCUS_REGIONS:
    # Find points near this region
    region_points = []  # (ring_idx, point_idx)
    for ri in range(len(simplified)):
        pts = simplified[ri]
        for pi in range(len(pts)):
            dx = (pts[pi, 0] - cx) / rx
            dy = (pts[pi, 1] - cy) / ry
            if dx*dx + dy*dy < 1.0:  # inside ellipse
                region_points.append((ri, pi))

    if not region_points:
        continue

    # Intensive perturbation on these points
    acc = 0; px_total = 0
    for _ in range(2000):
        ri, pi = region_points[rng.randint(len(region_points))]
        pts = simplified[ri]; n = len(pts)
        if n < 4: continue
        orig = ring_meta[ri]['original']; no = len(orig)
        _, base_idx = trees_f[ri].query(pts[pi])
        off = rng.randint(-PERTURB_SEARCH, PERTURB_SEARCH + 1)
        if off == 0: continue
        ci = (base_idx + off) % (no - 1)
        pos = orig[ci]
        if np.sqrt(((pts[pi] - pos)**2).sum()) < 0.05: continue
        d = multi_iou_delta(pts, [pi], [pos])
        if d > 0:
            simplified[ri][pi] = pos.copy()
            acc += 1; px_total += d

    if acc > 0:
        print(f"  {region_name}: {acc} moves, {px_total}px fixed ({len(region_points)} pts in region)")

iou, _, _, pred_mask = compute_iou_full(simplified, ring_meta)
print(f"\n  After focused optimization: IoU={iou:.4f} ({iou*100:.2f}%)")

total_dots = sum(len(s) for s in simplified)
print(f"\nFinal: {total_dots} dots, IoU={iou:.4f} ({iou*100:.2f}%)")

# Group by continent
continent_dots = {}
for ri, pts in enumerate(simplified):
    co = ring_meta[ri]['continent']
    if co not in continent_dots: continent_dots[co] = []
    continent_dots[co].append(pts)

for co in sorted(continent_dots.keys()):
    n = sum(len(d) for d in continent_dots[co])
    print(f"  {co}: {n} dots ({len(continent_dots[co])} outlines)")

# ============================================================
# 7. Baseline
# ============================================================
print("\nBaseline...")

def place_adaptive(rc, bs, mnf=0.4, mxf=3.5):
    n = len(rc)
    if n < 2: return np.array([])
    df = np.diff(rc, axis=0)
    sl = np.sqrt((df**2).sum(1))
    t = sl.sum()
    if t < bs: return np.array([])
    cm = np.zeros(n); cm[1:] = np.cumsum(sl)
    dr = np.arctan2(df[:, 1], df[:, 0])
    ac = np.zeros(n)
    for i in range(1, len(dr)):
        da = (dr[i]-dr[i-1]+np.pi)%(2*np.pi)-np.pi
        ac[i] = abs(da)
    w = max(3, n//50); k = np.ones(w)/w
    sm = np.convolve(ac, k, mode='same') if n > w else ac
    cx = np.percentile(sm[sm>0], 95) if np.any(sm>0) else 1.0
    if cx < 1e-6: cx = 1.0
    nm = np.clip(sm/cx, 0, 1)
    mn, mx = bs*mnf, bs*mxf
    def interp(d):
        i = np.searchsorted(cm, d, side='right')-1
        i = max(0, min(i, n-2))
        s = sl[i]; tt = np.clip((d-cm[i])/s, 0, 1) if s > 0 else 0
        return rc[i]+tt*(rc[i+1]-rc[i]), nm[i]*(1-tt)+nm[min(i+1,n-1)]*tt
    dots = []; d = 0.0
    while d < t:
        pt, c = interp(d); dots.append(pt)
        d += mx-(mx-mn)*c
    return np.array(dots) if len(dots) >= 3 else np.array([])

bsp = 3.5 / INCH_TO_CM
bd = []
bt = 0
bm = []
for ri, pd in enumerate(polygon_data):
    d = place_adaptive(pd['e'], bsp)
    if len(d) > 0:
        bd.append(d)
        bm.append({'ccw': True, 'continent': pd['c'], 'interiors': pd['i']})
        bt += len(d)

bi, bp_v, br_v, _ = compute_iou_full(bd, bm)
print(f"  Baseline: {bt} dots, IoU={bi:.4f} ({bi*100:.2f}%)")
print(f"\n  Improvement: {(iou-bi)*100:+.2f}%")

# ============================================================
# 8. Render
# ============================================================
print("\nStep 8: Rendering...")
from matplotlib.patches import Polygon as MplPolygon

FILL_PAL = {
    'Eurasia': ['#f4cccc','#e8b4b4','#f9d6d6','#fce5cd','#f0c4c4',
                '#ead1dc','#f5c6c6','#fdd9d9','#e6c1c1','#f2b8b8'],
    'Africa':  ['#fff2cc','#fce5b0','#f9ebc0','#ffd966','#ffe599',
                '#f6d78e','#ffeaaa','#f9dcaa','#ffe0a0','#f5d090'],
    'North America': ['#d9ead3','#b6d7a8','#c5e0b4','#a4c89a','#d3e8cb',
                      '#bfddb3','#cce5c0','#a9d49d','#c0dbb5','#b3d4a5'],
    'South America': ['#d9d2e9','#c4b8db','#e0d8f0','#b4a7d0','#cfc5e5',
                      '#d5cce8','#c8bfe0','#ddd5ee','#beb3d8','#d2c9e6'],
    'Oceania': ['#d0e8e4','#b5dbd5','#c2e0da','#a8d4cc','#bdddd6',
                '#c8e3dd','#aed7cf','#c5e1db','#b2d9d2','#bfded8'],
}

def extract_polys(g):
    r = []
    if isinstance(g, Polygon): r.append(np.array(g.exterior.coords))
    elif isinstance(g, MultiPolygon):
        for p in g.geoms: r.append(np.array(p.exterior.coords))
    return r

def proj_panel(c):
    cc = c.copy()
    if cc[:,0].max()-cc[:,0].min()>200: cc[cc[:,0]<0,0]+=360
    x,y = proj_miller(cc[:,0],cc[:,1])
    r = np.zeros_like(cc); r[:,0]=(x-xn)*sc+ox_off; r[:,1]=(y-yn)*sc+oy_off
    return r

fw = 24; fh = fw*PANEL_H/PANEL_W

def draw_map(ax, rings, meta, title):
    ax.set_xlim(0,PANEL_W); ax.set_ylim(0,PANEL_H)
    ax.set_aspect('equal'); ax.set_facecolor('#cce5ff')
    for ci,(cn,co,g) in enumerate(country_data):
        pl = FILL_PAL.get(co, FILL_PAL['Eurasia'])
        fc = pl[ci%len(pl)]
        for ec in extract_polys(g):
            try:
                pe = proj_panel(ec)
                if pe[:,0].max()<0 or pe[:,0].min()>PANEL_W: continue
                ax.add_patch(MplPolygon(pe,closed=True,facecolor=fc,edgecolor='#999',
                                        linewidth=0.1,alpha=0.8,zorder=1))
            except: continue
    cg = {}
    for ri,d in enumerate(rings):
        co = meta[ri]['continent']
        if co not in cg: cg[co]=[]
        cg[co].append(d)
    for co,rl in cg.items():
        cl = CONTINENT_COLORS.get(co,'#888')
        for d in rl:
            if len(d)<2: continue
            cs = np.vstack([d,d[0:1]])
            ax.plot(cs[:,0],cs[:,1],color=cl,linewidth=0.8,alpha=0.7,zorder=2)
            ax.scatter(d[:,0],d[:,1],s=6,c=cl,zorder=3,edgecolors='white',linewidths=0.2)
    ax.set_title(title,fontsize=10,fontweight='bold',pad=8)
    ax.set_xticks(np.arange(0,PANEL_W+1,10))
    ax.set_yticks(np.arange(0,PANEL_H+1,10))
    ax.tick_params(labelsize=6)

# Comparison
fig,axes = plt.subplots(1,2,figsize=(fw*2,fh),dpi=150)
draw_map(axes[0], simplified, ring_meta,
         f'v7 ({total_dots} dots, IoU={iou:.4f})')
# Need to build baseline meta for draw_map
draw_map(axes[1], bd, bm,
         f'Adaptive ({bt} dots, IoU={bi:.4f})')
plt.suptitle('v7 (Iterative Refinement) vs Adaptive',fontsize=13,fontweight='bold',y=0.98)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('world_map_IOU_v7_comparison.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print("Saved: world_map_IOU_v7_comparison.png")

# Standalone
fig2,ax2 = plt.subplots(1,1,figsize=(fw,fh),dpi=200)
draw_map(ax2, simplified, ring_meta,
         f'v7  |  {total_dots} dots  |  IoU={iou:.4f}')
handles = []
for co in sorted(continent_dots.keys()):
    cl = CONTINENT_COLORS.get(co,'#888')
    n = sum(len(d) for d in continent_dots[co])
    if n>0: handles.append(Line2D([0],[0],marker='o',color=cl,linewidth=1,markersize=6,
                                   markerfacecolor=cl,markeredgecolor='white',markeredgewidth=0.3,
                                   label=f'{co} ({n})'))
ax2.legend(handles=handles,loc='lower left',fontsize=8,framealpha=0.95)
plt.tight_layout()
plt.savefig('world_map_IOU_v7_greedy.png',dpi=200,bbox_inches='tight',facecolor='white')
plt.close()
print("Saved: world_map_IOU_v7_greedy.png")

# Heatmap
fig3,ax3 = plt.subplots(1,1,figsize=(fw,fh*0.8),dpi=150)
dm = np.zeros((RH,RW,3))
dm[land_mask&pred_mask]=[0.2,0.8,0.2]
dm[pred_mask&~land_mask]=[0.9,0.2,0.2]
dm[~pred_mask&land_mask]=[0.2,0.4,0.9]
dm[~pred_mask&~land_mask]=[0.9,0.95,1.0]
ax3.imshow(dm,origin='lower',extent=[0,PANEL_W,0,PANEL_H])
ax3.set_title(f'v7 Heatmap  |  IoU={iou:.4f}',fontsize=10,fontweight='bold')
ax3.set_xlabel('inch'); ax3.set_ylabel('inch')
plt.tight_layout()
plt.savefig('world_map_IOU_v7_heatmap.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close()
print("Saved: world_map_IOU_v7_heatmap.png")

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"v7 (Iterative Refine): {total_dots:>5} dots, IoU={iou:.4f} ({iou*100:.2f}%)")
print(f"Adaptive Spacing:      {bt:>5} dots, IoU={bi:.4f} ({bi*100:.2f}%)")
print(f"Improvement:           {(iou-bi)*100:+.2f}%")
print(f"{'='*60}")
