"""
World Map Spline Smoothing approach
1. Extract continent outlines
2. Simplify with DP
3. Chaikin corner-cutting smooth (2 iterations)
4. Resample uniformly along smooth curve
5. Compare IoU with greedy approach
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import shape, MultiPolygon, Polygon, LineString
from shapely.ops import unary_union

PANEL_W, PANEL_H = 72, 48
MARGIN = 2.0
DW, DH = PANEL_W - 2*MARGIN, PANEL_H - 2*MARGIN
INCH_TO_CM = 2.54
TARGET = 1200
MIN_PERIM = 3.0
PPI = 20
RW, RH = int(PANEL_W * PPI), int(PANEL_H * PPI)

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
print("Loading...")
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

# Project
def fix_am(c, co):
    cc = c.copy()
    if cc[:, 0].max() - cc[:, 0].min() > 200: cc[cc[:, 0] < 0, 0] += 360
    elif cc[:, 0].mean() < -100 and co == 'Eurasia': cc[:, 0] += 360
    return cc

def proj_ring(c, co):
    f = fix_am(c, co); x, y = proj_miller(f[:, 0], f[:, 1])
    return np.column_stack([x, y])

ap = []
for co, ps in cpoly.items():
    for p in ps: ap.append(proj_ring(np.array(p.exterior.coords), co))
pts = np.vstack(ap)
xn, yn = pts.min(0); xx, yx = pts.max(0)
xr, yr = xx - xn, yx - yn
sc = min(DW/xr, DH/yr); ox = MARGIN + (DW-xr*sc)/2; oy = MARGIN + (DH-yr*sc)/2

def to_panel(r):
    o = np.zeros_like(r); o[:,0]=(r[:,0]-xn)*sc+ox; o[:,1]=(r[:,1]-yn)*sc+oy; return o

polygon_data = []
tp = 0
for co, ps in cpoly.items():
    for p in ps:
        e = to_panel(proj_ring(np.array(p.exterior.coords), co))
        ii = [to_panel(proj_ring(np.array(i.coords), co)) for i in p.interiors]
        d = np.diff(e, axis=0); pm = np.sqrt((d**2).sum(1)).sum()
        if pm >= MIN_PERIM:
            polygon_data.append({'c': co, 'e': e, 'i': ii, 'p': pm})
            tp += pm

print(f"  {len(polygon_data)} polygons")

# Rasterize land
print(f"Rasterizing ({RW}x{RH})...")
land_mask = np.zeros((RH, RW), dtype=bool)

def rast(ring, mask, val=True):
    rx = ring[:,0]/PANEL_W*RW; ry = ring[:,1]/PANEL_H*RH
    path = Path(np.column_stack([rx, ry]))
    y0=max(0,int(np.floor(ry.min()))); y1=min(RH,int(np.ceil(ry.max()))+1)
    x0=max(0,int(np.floor(rx.min()))); x1=min(RW,int(np.ceil(rx.max()))+1)
    if x1<=x0 or y1<=y0: return
    yy,xx=np.mgrid[y0:y1,x0:x1]; pts=np.column_stack([xx.ravel(),yy.ravel()])
    m=path.contains_points(pts).reshape(yy.shape)
    if val: mask[y0:y1,x0:x1]|=m
    else: mask[y0:y1,x0:x1]&=~m

for pd in polygon_data:
    rast(pd['e'], land_mask, True)
    for i in pd['i']: rast(i, land_mask, False)

print(f"  Land: {land_mask.sum()} px")

# ============================================================
# Chaikin corner-cutting smoothing
# ============================================================
def chaikin_smooth(pts, iterations=2):
    """Chaikin corner-cutting for closed polygon."""
    for _ in range(iterations):
        n = len(pts)
        new_pts = []
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts.append(0.75 * p0 + 0.25 * p1)
            new_pts.append(0.25 * p0 + 0.75 * p1)
        pts = np.array(new_pts)
    return pts

def resample_ring(pts, n_points):
    """Resample a closed ring to exactly n_points, uniformly by arc length."""
    # Close the ring
    closed = np.vstack([pts, pts[0:1]])
    diffs = np.diff(closed, axis=0)
    seg_lens = np.sqrt((diffs**2).sum(axis=1))
    total = seg_lens.sum()
    if total < 1e-6: return pts[:n_points] if len(pts) >= n_points else pts

    cum = np.zeros(len(closed))
    cum[1:] = np.cumsum(seg_lens)

    result = []
    for i in range(n_points):
        target = total * i / n_points
        idx = np.searchsorted(cum, target, side='right') - 1
        idx = max(0, min(idx, len(closed) - 2))
        sl = seg_lens[idx]
        t = np.clip((target - cum[idx]) / sl, 0, 1) if sl > 0 else 0
        pt = closed[idx] + t * (closed[idx+1] - closed[idx])
        result.append(pt)
    return np.array(result)

# ============================================================
# Pipeline: DP simplify -> Chaikin smooth -> Resample
# ============================================================
print("\nSpline pipeline...")

def dp_to_n(ring, target_n):
    if len(ring) <= target_n: return ring.copy()
    line = LineString(ring)
    d = np.diff(ring, axis=0); sl = np.sqrt((d**2).sum(1))
    mx = sl.sum()/4; mn = max(1e-8, sl[sl>0].min()/10)
    best = ring.copy(); bd = len(ring)
    for _ in range(30):
        mid = (mn+mx)/2; s = line.simplify(mid, preserve_topology=True); n = len(s.coords)
        if abs(n-target_n) < bd: bd = abs(n-target_n); best = np.array(s.coords)
        if n > target_n: mn = mid
        elif n < target_n: mx = mid
        else: break
    return best

# DP to 3x target, then smooth, then resample to target
DP_MULT = 3
simplified_data = []
total_dots = 0

for pd in polygon_data:
    ext = pd['e']
    perim = pd['p']
    budget = max(4, int(round(TARGET * perim / tp)))

    # Step 1: DP simplify to ~3x budget
    dp_target = min(len(ext), budget * DP_MULT)
    dp_pts = dp_to_n(ext, dp_target)

    # Remove closing duplicate if present
    if len(dp_pts) > 1 and np.allclose(dp_pts[0], dp_pts[-1]):
        dp_pts = dp_pts[:-1]

    # Step 2: Chaikin smooth (2 iterations)
    smoothed = chaikin_smooth(dp_pts, iterations=2)

    # Step 3: Resample to budget
    resampled = resample_ring(smoothed, budget)

    simplified_data.append({
        'continent': pd['c'],
        'exterior': resampled,
        'interiors': pd['i'],
    })
    total_dots += len(resampled)

print(f"  Total dots: {total_dots}")

# ============================================================
# Compute IoU
# ============================================================
print("Computing IoU...")

def compute_iou(sl, lm):
    pm = np.zeros_like(lm)
    for sd in sl:
        e = sd['exterior']
        if len(e) < 3: continue
        cl = np.vstack([e, e[0:1]])
        rx = cl[:,0]/PANEL_W*RW; ry = cl[:,1]/PANEL_H*RH
        p = Path(np.column_stack([rx, ry]))
        y0=max(0,int(np.floor(ry.min()))); y1=min(RH,int(np.ceil(ry.max()))+1)
        x0=max(0,int(np.floor(rx.min()))); x1=min(RW,int(np.ceil(rx.max()))+1)
        if x1<=x0 or y1<=y0: continue
        yy,xx=np.mgrid[y0:y1,x0:x1]; pts=np.column_stack([xx.ravel(),yy.ravel()])
        m=p.contains_points(pts).reshape(yy.shape); pm[y0:y1,x0:x1]|=m
        for it in sd.get('interiors', []):
            if len(it) < 3: continue
            ci=np.vstack([it,it[0:1]])
            rxi=ci[:,0]/PANEL_W*RW; ryi=ci[:,1]/PANEL_H*RH
            pi=Path(np.column_stack([rxi,ryi]))
            ym=max(0,int(np.floor(ryi.min()))); yM=min(RH,int(np.ceil(ryi.max()))+1)
            xm=max(0,int(np.floor(rxi.min()))); xM=min(RW,int(np.ceil(rxi.max()))+1)
            if xM<=xm or yM<=ym: continue
            yy2,xx2=np.mgrid[ym:yM,xm:xM]; p2=np.column_stack([xx2.ravel(),yy2.ravel()])
            m2=pi.contains_points(p2).reshape(yy2.shape); pm[ym:yM,xm:xM]&=~m2
    inter=(pm&lm).sum(); union=(pm|lm).sum()
    iou=inter/union if union>0 else 0
    prec=inter/pm.sum() if pm.sum()>0 else 0
    rec=inter/lm.sum() if lm.sum()>0 else 0
    return iou, prec, rec, pm

iou, prec, rec, pred_mask = compute_iou(simplified_data, land_mask)
print(f"  Spline: {total_dots} dots, IoU={iou:.4f} ({iou*100:.2f}%)")
print(f"  Precision={prec:.4f}, Recall={rec:.4f}")

# Check self-intersections
from shapely.geometry import LinearRing
n_cross = 0
for sd in simplified_data:
    e = sd['exterior']
    if len(e) < 4: continue
    lr = LinearRing(e)
    if not lr.is_simple:
        n_cross += 1
print(f"  Self-intersecting rings: {n_cross}")

# ============================================================
# Render
# ============================================================
print("\nRendering...")

FILL_PAL = {
    'Eurasia': ['#f4cccc','#e8b4b4','#f9d6d6','#fce5cd','#f0c4c4'],
    'Africa':  ['#fff2cc','#fce5b0','#f9ebc0','#ffd966','#ffe599'],
    'North America': ['#d9ead3','#b6d7a8','#c5e0b4','#a4c89a','#d3e8cb'],
    'South America': ['#d9d2e9','#c4b8db','#e0d8f0','#b4a7d0','#cfc5e5'],
    'Oceania': ['#d0e8e4','#b5dbd5','#c2e0da','#a8d4cc','#bdddd6'],
}

def extract_polys(g):
    r = []
    if isinstance(g, Polygon): r.append(np.array(g.exterior.coords))
    elif isinstance(g, MultiPolygon):
        for p in g.geoms: r.append(np.array(p.exterior.coords))
    return r

def proj_panel(c):
    cc=c.copy()
    if cc[:,0].max()-cc[:,0].min()>200: cc[cc[:,0]<0,0]+=360
    x,y=proj_miller(cc[:,0],cc[:,1])
    r=np.zeros_like(cc); r[:,0]=(x-xn)*sc+ox; r[:,1]=(y-yn)*sc+oy
    return r

fw=24; fh=fw*PANEL_H/PANEL_W

fig, ax = plt.subplots(1, 1, figsize=(fw, fh), dpi=200)
ax.set_xlim(0, PANEL_W); ax.set_ylim(0, PANEL_H)
ax.set_aspect('equal'); ax.set_facecolor('#cce5ff')

for ci, (cn, co, g) in enumerate(country_data):
    pl = FILL_PAL.get(co, FILL_PAL['Eurasia'])
    fc = pl[ci % len(pl)]
    for ec in extract_polys(g):
        try:
            pe = proj_panel(ec)
            if pe[:,0].max()<0 or pe[:,0].min()>PANEL_W: continue
            ax.add_patch(MplPolygon(pe, closed=True, facecolor=fc,
                                    edgecolor='#999', linewidth=0.1, alpha=0.8, zorder=1))
        except: continue

continent_dots = {}
for sd in simplified_data:
    co = sd['continent']
    if co not in continent_dots: continent_dots[co] = []
    continent_dots[co].append(sd['exterior'])

for co, rl in continent_dots.items():
    cl = CONTINENT_COLORS.get(co, '#888')
    for d in rl:
        if len(d) < 2: continue
        cs = np.vstack([d, d[0:1]])
        ax.plot(cs[:,0], cs[:,1], color=cl, linewidth=0.8, alpha=0.7, zorder=2)
        ax.scatter(d[:,0], d[:,1], s=6, c=cl, zorder=3, edgecolors='white', linewidths=0.2)

handles = []
for co in sorted(continent_dots.keys()):
    cl = CONTINENT_COLORS.get(co, '#888')
    n = sum(len(d) for d in continent_dots[co])
    if n > 0:
        handles.append(Line2D([0],[0], marker='o', color=cl, linewidth=1, markersize=6,
                              markerfacecolor=cl, markeredgecolor='white', markeredgewidth=0.3,
                              label=f'{co} ({n})'))
ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.95)
ax.set_title(f'Spline Smooth  |  {total_dots} dots  |  IoU={iou:.4f}  |  '
             f'Self-cross: {n_cross}', fontsize=10, fontweight='bold', pad=8)
ax.set_xticks(np.arange(0, PANEL_W+1, 10))
ax.set_yticks(np.arange(0, PANEL_H+1, 10))
ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig('world_map_SPLINE.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_SPLINE.png")

# Heatmap
fig2, ax2 = plt.subplots(1, 1, figsize=(fw, fh*0.8), dpi=150)
dm = np.zeros((RH, RW, 3))
dm[land_mask & pred_mask] = [0.2, 0.8, 0.2]
dm[pred_mask & ~land_mask] = [0.9, 0.2, 0.2]
dm[~pred_mask & land_mask] = [0.2, 0.4, 0.9]
dm[~pred_mask & ~land_mask] = [0.9, 0.95, 1.0]
ax2.imshow(dm, origin='lower', extent=[0, PANEL_W, 0, PANEL_H])
ax2.set_title(f'Spline Heatmap  |  IoU={iou:.4f}', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('world_map_SPLINE_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_SPLINE_heatmap.png")

print(f"\n{'='*50}")
print(f"Spline: {total_dots} dots, IoU={iou:.4f} ({iou*100:.2f}%)")
print(f"Self-intersections: {n_cross}")
print(f"{'='*50}")
