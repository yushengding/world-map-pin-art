"""
World Map Final Version - Miller Projection, 1052 dots
Eurasia merged, antimeridian fixed, continent-colored dots+lines
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import shape, MultiPolygon, Polygon, box
from shapely.ops import unary_union

# ============================================================
# CONFIG
# ============================================================
PANEL_WIDTH_INCH = 72
PANEL_HEIGHT_INCH = 48
MARGIN_INCH = 2.0
DOT_SPACING_CM = 3.5
INCH_TO_CM = 2.54
DOT_SPACING_INCH = DOT_SPACING_CM / INCH_TO_CM
DRAW_WIDTH = PANEL_WIDTH_INCH - 2 * MARGIN_INCH
DRAW_HEIGHT = PANEL_HEIGHT_INCH - 2 * MARGIN_INCH

CONTINENT_COLORS = {
    'Eurasia':       '#e74c3c',
    'Africa':        '#f39c12',
    'North America': '#2ecc71',
    'South America': '#9b59b6',
    'Oceania':       '#1abc9c',
}

# Country -> Continent
CONTINENT_MAP = {}
_ASIA = [
    'AFG','ARM','AZE','BHR','BGD','BTN','BRN','KHM','CHN','CYP','GEO','IND',
    'IDN','IRN','IRQ','ISR','JPN','JOR','KAZ','KWT','KGZ','LAO','LBN','MYS',
    'MDV','MNG','MMR','NPL','OMN','PAK','PSE','PHL','QAT','SAU','SGP','KOR',
    'PRK','LKA','SYR','TWN','TJK','THA','TLS','TKM','ARE','UZB','VNM',
    'YEM','IOT','CCK','CXR','HKG','MAC','RUS','TUR',
]
_EUROPE = [
    'ALB','AND','AUT','BLR','BEL','BIH','BGR','HRV','CZE','DNK','EST','FIN',
    'FRA','DEU','GRC','HUN','ISL','IRL','ITA','XKX','LVA','LIE','LTU','LUX',
    'MLT','MDA','MCO','MNE','NLD','MKD','NOR','POL','PRT','ROU','SMR',
    'SRB','SVK','SVN','ESP','SWE','CHE','UKR','GBR','VAT','FRO','GGY','IMN',
    'JEY','ALA','GIB','SJM','AKR',
]
_AFRICA = [
    'DZA','AGO','BEN','BWA','BFA','BDI','CPV','CMR','CAF','TCD','COM','COG',
    'COD','CIV','DJI','EGY','GNQ','ERI','SWZ','ETH','GAB','GMB','GHA','GIN',
    'GNB','KEN','LSO','LBR','LBY','MDG','MWI','MLI','MRT','MUS','MAR','MOZ',
    'NAM','NER','NGA','RWA','STP','SEN','SYC','SLE','SOM','ZAF','SSD','SDN',
    'TZA','TGO','TUN','UGA','ZMB','ZWE','SOL','MYT','REU','SHN','ESH','BIR',
]
_NORTH_AMERICA = [
    'ATG','BHS','BRB','BLZ','CAN','CRI','CUB','DMA','DOM','SLV','GRD','GTM',
    'HTI','HND','JAM','MEX','NIC','PAN','KNA','LCA','VCT','TTO','USA',
    'AIA','ABW','BMU','VGB','CYM','CUW','GLP','MTQ','MSR','PRI','BES',
    'SXM','MAF','TCA','VIR','GRL','SPM','CLP',
]
_SOUTH_AMERICA = [
    'ARG','BOL','BRA','CHL','COL','ECU','GUY','PRY','PER','SUR','URY','VEN',
    'GUF','FLK','SGS','BVT','BRI',
]
_OCEANIA = [
    'AUS','FJI','KIR','MHL','FSM','NRU','NZL','PLW','PNG','WSM','SLB','TON',
    'TUV','VUT','ASM','COK','PYF','GUM','MNP','NCL','NIU','NFK','PCN','TKL',
    'UMI','WLF','ACI','ATC','CSI',
]
for codes, cont in [
    (_ASIA, 'Asia'), (_EUROPE, 'Europe'), (_AFRICA, 'Africa'),
    (_NORTH_AMERICA, 'North America'), (_SOUTH_AMERICA, 'South America'),
    (_OCEANIA, 'Oceania'), (['ATA'], 'Antarctica'),
]:
    for c in codes:
        CONTINENT_MAP[c] = cont

def guess_continent(lon, lat):
    if lat < -60: return 'Antarctica'
    if lat > 60 and lon < -10: return 'North America'
    if lon < -30: return 'North America' if lat > 15 else 'South America'
    if lon < 60: return 'Europe' if lat > 35 else 'Africa'
    if lon < 150: return 'Asia'
    return 'Oceania'

# ============================================================
# Miller projection
# ============================================================
def proj_miller(lon, lat):
    lat_r = np.radians(np.clip(lat, -85, 85))
    y = np.degrees(1.25 * np.log(np.tan(np.pi/4 + 0.4 * lat_r)))
    return lon.copy(), y

# ============================================================
# Load & process
# ============================================================
print("Loading world map...")
with open('countries.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

def extract_rings(geom):
    rings = []
    if isinstance(geom, Polygon):
        rings.append(np.array(geom.exterior.coords))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            rings.append(np.array(poly.exterior.coords))
    return rings

# Per-country data for detailed colored map
country_data = []  # list of (name, continent, geom)

continent_geoms = {}
for feature in data['features']:
    props = feature['properties']
    iso3 = props.get('ISO3166-1-Alpha-3', '')
    name = props.get('name', 'Unknown')
    geom = shape(feature['geometry'])
    if not geom.is_valid:
        geom = geom.buffer(0)

    continent = CONTINENT_MAP.get(iso3)
    if not continent:
        continent = guess_continent(geom.centroid.x, geom.centroid.y)
    if continent == 'Antarctica':
        continue

    # Map Europe/Asia -> Eurasia for consistency
    display_cont = 'Eurasia' if continent in ('Europe', 'Asia') else continent
    country_data.append((name, display_cont, geom))

    if continent not in continent_geoms:
        continent_geoms[continent] = []
    continent_geoms[continent].append(geom)

# Merge Europe + Asia -> Eurasia
eurasia_geoms = []
for key in ['Europe', 'Asia']:
    if key in continent_geoms:
        eurasia_geoms.extend(continent_geoms.pop(key))
if eurasia_geoms:
    continent_geoms['Eurasia'] = eurasia_geoms

print("Loading outlines...")

# Merge per continent & extract rings
continent_rings = {}
for continent, geoms in continent_geoms.items():
    merged = unary_union(geoms)

    rings = extract_rings(merged)
    # Fix antimeridian: shift negative-lon parts to positive side
    fixed = []
    for ring in rings:
        lon_span = ring[:, 0].max() - ring[:, 0].min()
        lon_mean = ring[:, 0].mean()
        if lon_span > 200:
            r = ring.copy()
            r[r[:, 0] < 0, 0] += 360
            fixed.append(r)
        elif lon_mean < -100 and continent == 'Eurasia':
            r = ring.copy()
            r[:, 0] += 360
            fixed.append(r)
        else:
            fixed.append(ring)
    continent_rings[continent] = fixed

print("Continents:", list(continent_rings.keys()))

# ============================================================
# Project & scale
# ============================================================
all_proj = []
continent_proj = {}
for cont, rings in continent_rings.items():
    proj_rings = []
    for ring in rings:
        x, y = proj_miller(ring[:, 0], ring[:, 1])
        proj_rings.append(np.column_stack([x, y]))
        all_proj.append(proj_rings[-1])
    continent_proj[cont] = proj_rings

pts = np.vstack(all_proj)
xn, yn = pts.min(axis=0)
xx, yx = pts.max(axis=0)
xr, yr = xx - xn, yx - yn
sc = min(DRAW_WIDTH / xr, DRAW_HEIGHT / yr)
ox = MARGIN_INCH + (DRAW_WIDTH - xr * sc) / 2
oy = MARGIN_INCH + (DRAW_HEIGHT - yr * sc) / 2

def to_panel(ring):
    r = np.zeros_like(ring)
    r[:, 0] = (ring[:, 0] - xn) * sc + ox
    r[:, 1] = (ring[:, 1] - yn) * sc + oy
    return r

continent_panel = {}
for cont, rings in continent_proj.items():
    continent_panel[cont] = [to_panel(r) for r in rings]

# Keep raw projected data for reference overlay
continent_panel_raw = {c: [r.copy() for r in rings] for c, rings in continent_panel.items()}

# ============================================================
# Adaptive dot placement: dense on curves, sparse on straights
# All outlines remain CLOSED loops
# ============================================================
def place_dots_adaptive(ring_coords, base_spacing,
                        min_spacing_factor=0.4, max_spacing_factor=3.5):
    """Curvature-adaptive dot placement. Returns closed-loop points."""
    n = len(ring_coords)
    if n < 2: return np.array([])

    diffs = np.diff(ring_coords, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    total = seg_lengths.sum()
    if total < base_spacing * 0.5: return np.array([])

    cum = np.zeros(n)
    cum[1:] = np.cumsum(seg_lengths)

    # Direction angles
    directions = np.arctan2(diffs[:, 1], diffs[:, 0])

    # Curvature: angle change between consecutive segments
    angle_changes = np.zeros(n)
    for i in range(1, len(directions)):
        da = directions[i] - directions[i-1]
        da = (da + np.pi) % (2 * np.pi) - np.pi
        angle_changes[i] = abs(da)

    # Smooth curvature
    window = max(3, n // 50)
    if n > window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(angle_changes, kernel, mode='same')
    else:
        smoothed = angle_changes

    curv_max = np.percentile(smoothed[smoothed > 0], 95) if np.any(smoothed > 0) else 1.0
    if curv_max < 1e-6: curv_max = 1.0
    norm_curv = np.clip(smoothed / curv_max, 0, 1)

    min_sp = base_spacing * min_spacing_factor
    max_sp = base_spacing * max_spacing_factor

    def interp(dist):
        idx = np.searchsorted(cum, dist, side='right') - 1
        idx = max(0, min(idx, n - 2))
        sl = seg_lengths[idx]
        t = np.clip((dist - cum[idx]) / sl, 0, 1) if sl > 0 else 0
        pt = ring_coords[idx] + t * (ring_coords[idx + 1] - ring_coords[idx])
        curv = norm_curv[idx] * (1 - t) + norm_curv[min(idx + 1, n - 1)] * t
        return pt, curv

    dots = []
    dist = 0.0
    while dist < total:
        pt, curv = interp(dist)
        dots.append(pt)
        step = max_sp - (max_sp - min_sp) * curv
        dist += step

    if len(dots) < 3:
        return np.array([])
    return np.array(dots)

MIN_PERIM = DOT_SPACING_INCH * 3

continent_dots = {}
total_dots = 0
for cont, rings in continent_panel.items():
    ring_dots = []
    for ring in rings:
        diffs = np.diff(ring, axis=0)
        perim = np.sqrt((diffs**2).sum(axis=1)).sum()
        if perim < MIN_PERIM:
            continue
        dots = place_dots_adaptive(ring, DOT_SPACING_INCH)
        if len(dots) > 0:
            ring_dots.append(dots)
            total_dots += len(dots)
    continent_dots[cont] = ring_dots

def maybe_close(dots, max_edge=999):
    """Always close the loop - these are continent outlines."""
    return np.vstack([dots, dots[0:1]])

print(f"\nTotal dots: {total_dots}")
for cont in sorted(continent_dots.keys()):
    n = sum(len(d) for d in continent_dots[cont])
    print(f"  {cont}: {n} dots ({len(continent_dots[cont])} outlines)")

# ============================================================
# Render final large map
# ============================================================
print("\nRendering final map...")

# Use actual panel proportions for figure
fig_w = 24
fig_h = fig_w * PANEL_HEIGHT_INCH / PANEL_WIDTH_INCH
fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=250)

ax.set_xlim(0, PANEL_WIDTH_INCH)
ax.set_ylim(0, PANEL_HEIGHT_INCH)
ax.set_aspect('equal')
ax.set_facecolor('white')

# Panel border
panel_rect = mpatches.Rectangle(
    (0, 0), PANEL_WIDTH_INCH, PANEL_HEIGHT_INCH,
    linewidth=1.5, edgecolor='#333333', facecolor='none'
)
ax.add_patch(panel_rect)

# Drawing area border (margin indicator)
draw_rect = mpatches.Rectangle(
    (MARGIN_INCH, MARGIN_INCH), DRAW_WIDTH, DRAW_HEIGHT,
    linewidth=0.5, edgecolor='#cccccc', facecolor='none', linestyle=':'
)
ax.add_patch(draw_rect)

# Draw continent outlines
for cont, ring_dots in continent_dots.items():
    color = CONTINENT_COLORS.get(cont, '#888888')
    for dots in ring_dots:
        if len(dots) < 2:
            continue
        closed = maybe_close(dots)
        # Lines
        ax.plot(closed[:, 0], closed[:, 1],
                color=color, linewidth=0.7, alpha=0.5, zorder=1)
        # Dots
        ax.scatter(dots[:, 0], dots[:, 1],
                   s=8, c=color, zorder=2, edgecolors='white', linewidths=0.3)

# Legend
legend_handles = []
for cont in ['Eurasia', 'Africa', 'North America', 'South America', 'Oceania']:
    if cont in continent_dots and continent_dots[cont]:
        n = sum(len(d) for d in continent_dots[cont])
        color = CONTINENT_COLORS[cont]
        legend_handles.append(
            Line2D([0], [0], marker='o', color=color, linewidth=1,
                   markersize=7, markerfacecolor=color, markeredgecolor='white',
                   markeredgewidth=0.4, label=f'{cont} ({n})')
        )

ax.legend(handles=legend_handles, loc='lower left', fontsize=9,
          framealpha=0.95, edgecolor='#cccccc', fancybox=True,
          borderpad=1, labelspacing=0.8)

# Title
ax.set_title(
    f'World Map Dot Outline  |  Miller Projection  |  '
    f'{PANEL_WIDTH_INCH} x {PANEL_HEIGHT_INCH}" ({PANEL_WIDTH_INCH*INCH_TO_CM:.0f} x {PANEL_HEIGHT_INCH*INCH_TO_CM:.0f}cm)  |  '
    f'Dot spacing: {DOT_SPACING_CM}cm  |  Total: {total_dots} dots',
    fontsize=12, pad=15, fontweight='bold'
)

# Axis labels with inch markings
ax.set_xlabel('inch', fontsize=10)
ax.set_ylabel('inch', fontsize=10)
ax.set_xticks(np.arange(0, PANEL_WIDTH_INCH + 1, 5))
ax.set_yticks(np.arange(0, PANEL_HEIGHT_INCH + 1, 5))
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.1, linewidth=0.3)

# Scale bar (4cm reference)
bar_x, bar_y = 3, 3
ax.plot([bar_x, bar_x + DOT_SPACING_INCH], [bar_y, bar_y],
        color='#333333', linewidth=2, zorder=5)
ax.plot([bar_x, bar_x], [bar_y - 0.2, bar_y + 0.2],
        color='#333333', linewidth=1.5, zorder=5)
ax.plot([bar_x + DOT_SPACING_INCH, bar_x + DOT_SPACING_INCH],
        [bar_y - 0.2, bar_y + 0.2],
        color='#333333', linewidth=1.5, zorder=5)
ax.text(bar_x + DOT_SPACING_INCH / 2, bar_y + 0.5,
        f'{DOT_SPACING_CM}cm ({DOT_SPACING_INCH:.2f}")',
        ha='center', fontsize=8, color='#333333', fontweight='bold')

plt.tight_layout()
plt.savefig('world_map_FINAL.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_FINAL.png")

# ============================================================
# Also render a clean print-ready version (no grid, no axis)
# ============================================================
print("Rendering print-ready version...")

fig2, ax2 = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=250)
ax2.set_xlim(0, PANEL_WIDTH_INCH)
ax2.set_ylim(0, PANEL_HEIGHT_INCH)
ax2.set_aspect('equal')
ax2.set_facecolor('white')

# Panel border only
panel_rect2 = mpatches.Rectangle(
    (0, 0), PANEL_WIDTH_INCH, PANEL_HEIGHT_INCH,
    linewidth=1.5, edgecolor='#333333', facecolor='none'
)
ax2.add_patch(panel_rect2)

for cont, ring_dots in continent_dots.items():
    color = CONTINENT_COLORS.get(cont, '#888888')
    for dots in ring_dots:
        if len(dots) < 2:
            continue
        closed = maybe_close(dots)
        ax2.plot(closed[:, 0], closed[:, 1],
                 color=color, linewidth=0.9, alpha=0.5, zorder=1)
        ax2.scatter(dots[:, 0], dots[:, 1],
                    s=10, c=color, zorder=2, edgecolors='white', linewidths=0.3)

ax2.legend(handles=legend_handles, loc='lower left', fontsize=9,
           framealpha=0.95, edgecolor='#cccccc', fancybox=True,
           borderpad=1, labelspacing=0.8)

ax2.axis('off')
plt.tight_layout()
plt.savefig('world_map_FINAL_clean.png', dpi=250, bbox_inches='tight',
            facecolor='white', pad_inches=0.1)
plt.close()
print("Saved: world_map_FINAL_clean.png")

# ============================================================
# Overlay version: real coastline + dots/lines
# Shows where dots simplify vs actual coastline
# ============================================================
print("Rendering overlay comparison version...")

fig_ov, ax_ov = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=250)
ax_ov.set_xlim(0, PANEL_WIDTH_INCH)
ax_ov.set_ylim(0, PANEL_HEIGHT_INCH)
ax_ov.set_aspect('equal')
ax_ov.set_facecolor('white')

panel_rect_ov = mpatches.Rectangle(
    (0, 0), PANEL_WIDTH_INCH, PANEL_HEIGHT_INCH,
    linewidth=1.5, edgecolor='#333333', facecolor='none'
)
ax_ov.add_patch(panel_rect_ov)

# Layer 1: Real coastline outlines (light gray, all detail)
for cont, rings in continent_panel.items():
    cont_color = CONTINENT_COLORS.get(cont, '#888888')
    # Use a lighter version of the continent color
    for ring in rings:
        diffs = np.diff(ring, axis=0)
        perim = np.sqrt((diffs**2).sum(axis=1)).sum()
        if perim < 1.0:  # skip tiny specks
            continue
        ax_ov.plot(ring[:, 0], ring[:, 1],
                   color=cont_color, linewidth=0.4, alpha=0.25, zorder=1)

# Layer 2: Dot-connected lines (solid color)
for cont, ring_dots in continent_dots.items():
    color = CONTINENT_COLORS.get(cont, '#888888')
    for dots in ring_dots:
        if len(dots) < 2:
            continue
        closed = maybe_close(dots)
        ax_ov.plot(closed[:, 0], closed[:, 1],
                   color=color, linewidth=0.8, alpha=0.6, zorder=2)
        ax_ov.scatter(dots[:, 0], dots[:, 1],
                      s=8, c=color, zorder=3, edgecolors='white', linewidths=0.3)

# Legend
overlay_handles = []
for cont in ['Eurasia', 'Africa', 'North America', 'South America', 'Oceania']:
    if cont in continent_dots and continent_dots[cont]:
        n = sum(len(d) for d in continent_dots[cont])
        color = CONTINENT_COLORS[cont]
        overlay_handles.append(
            Line2D([0], [0], color=color, linewidth=0.4, alpha=0.3,
                   label=f'{cont} actual coastline'))
        overlay_handles.append(
            Line2D([0], [0], marker='o', color=color, linewidth=1,
                   markersize=5, markerfacecolor=color, markeredgecolor='white',
                   markeredgewidth=0.3, label=f'{cont} dots ({n})'))

ax_ov.legend(handles=overlay_handles, loc='lower left', fontsize=7,
             framealpha=0.95, edgecolor='#cccccc', fancybox=True,
             borderpad=1, labelspacing=0.5, ncol=2)

ax_ov.set_title(
    f'Overlay: Actual Coastline vs Dot Outline  |  {total_dots} dots @ {DOT_SPACING_CM}cm',
    fontsize=11, pad=10, fontweight='bold'
)
ax_ov.set_xlabel('inch', fontsize=10)
ax_ov.set_ylabel('inch', fontsize=10)
ax_ov.set_xticks(np.arange(0, PANEL_WIDTH_INCH + 1, 5))
ax_ov.set_yticks(np.arange(0, PANEL_HEIGHT_INCH + 1, 5))
ax_ov.tick_params(labelsize=7)
ax_ov.grid(True, alpha=0.08, linewidth=0.2)

plt.tight_layout()
plt.savefig('world_map_OVERLAY.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_OVERLAY.png")

# ============================================================
# Zoomed overlay views for key regions
# ============================================================
print("Rendering zoomed overlay views...")

zoom_regions = {
    'Europe': (28, 46, 26, 40),
    'East Asia': (52, 70, 24, 42),
    'Americas': (2, 28, 8, 40),
}

fig_z, axes_z = plt.subplots(1, 3, figsize=(24, 10), dpi=200)

for ax_i, (region_name, (x1, x2, y1, y2)) in enumerate(zoom_regions.items()):
    ax = axes_z[ax_i]
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.set_aspect('equal')
    ax.set_facecolor('#faf8f2')

    # Real coastline
    for cont, rings in continent_panel.items():
        cont_color = CONTINENT_COLORS.get(cont, '#888888')
        for ring in rings:
            diffs = np.diff(ring, axis=0)
            perim = np.sqrt((diffs**2).sum(axis=1)).sum()
            if perim < 1.0:
                continue
            ax.plot(ring[:, 0], ring[:, 1],
                    color=cont_color, linewidth=0.6, alpha=0.3, zorder=1)

    # Dots + lines
    for cont, ring_dots in continent_dots.items():
        color = CONTINENT_COLORS.get(cont, '#888888')
        for dots in ring_dots:
            if len(dots) < 2:
                continue
            closed = maybe_close(dots)
            ax.plot(closed[:, 0], closed[:, 1],
                    color=color, linewidth=1.2, alpha=0.7, zorder=2)
            ax.scatter(dots[:, 0], dots[:, 1],
                       s=20, c=color, zorder=3, edgecolors='white', linewidths=0.4)

    ax.set_title(f'{region_name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('inch', fontsize=8)
    ax.set_ylabel('inch', fontsize=8)
    ax.grid(True, alpha=0.12, linewidth=0.3)

plt.suptitle(
    'Zoomed Overlay: Actual Coastline (light) vs Dot Outline (solid)',
    fontsize=12, fontweight='bold', y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('world_map_OVERLAY_zoom.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_OVERLAY_zoom.png")

# ============================================================
# Detailed colored world map + dots/lines overlay
# ============================================================
print("Rendering detailed colored map with dots overlay...")

from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# Color palettes for countries (varied within each continent)
CONTINENT_FILL_PALETTES = {
    'Eurasia': ['#f4cccc', '#e8b4b4', '#f9d6d6', '#fce5cd', '#f0c4c4',
                '#ead1dc', '#f5c6c6', '#fdd9d9', '#e6c1c1', '#f2b8b8'],
    'Africa':  ['#fff2cc', '#fce5b0', '#f9ebc0', '#ffd966', '#ffe599',
                '#f6d78e', '#ffeaaa', '#f9dcaa', '#ffe0a0', '#f5d090'],
    'North America': ['#d9ead3', '#b6d7a8', '#c5e0b4', '#a4c89a', '#d3e8cb',
                      '#bfddb3', '#cce5c0', '#a9d49d', '#c0dbb5', '#b3d4a5'],
    'South America': ['#d9d2e9', '#c4b8db', '#e0d8f0', '#b4a7d0', '#cfc5e5',
                      '#d5cce8', '#c8bfe0', '#ddd5ee', '#beb3d8', '#d2c9e6'],
    'Oceania': ['#d0e8e4', '#b5dbd5', '#c2e0da', '#a8d4cc', '#bdddd6',
                '#c8e3dd', '#aed7cf', '#c5e1db', '#b2d9d2', '#bfded8'],
}

def extract_polys(geom):
    """Extract list of (exterior_coords, [interior_coords]) from geometry."""
    polys = []
    if isinstance(geom, Polygon):
        polys.append((np.array(geom.exterior.coords),
                       [np.array(h.coords) for h in geom.interiors]))
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            polys.append((np.array(p.exterior.coords),
                           [np.array(h.coords) for h in p.interiors]))
    return polys

def proj_and_panel(coords):
    """Project lon/lat coords to panel coordinates."""
    # Fix antimeridian
    c = coords.copy()
    lon_span = c[:, 0].max() - c[:, 0].min()
    if lon_span > 200:
        c[c[:, 0] < 0, 0] += 360
    x, y = proj_miller(c[:, 0], c[:, 1])
    result = np.zeros_like(c)
    result[:, 0] = (x - xn) * sc + ox
    result[:, 1] = (y - yn) * sc + oy
    return result

# Full map with colored countries
fig_cm, ax_cm = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=250)
ax_cm.set_xlim(0, PANEL_WIDTH_INCH)
ax_cm.set_ylim(0, PANEL_HEIGHT_INCH)
ax_cm.set_aspect('equal')
ax_cm.set_facecolor('#cce5ff')  # Ocean blue

# Panel border
panel_rect_cm = mpatches.Rectangle(
    (0, 0), PANEL_WIDTH_INCH, PANEL_HEIGHT_INCH,
    linewidth=1.5, edgecolor='#333333', facecolor='none', zorder=10
)
ax_cm.add_patch(panel_rect_cm)

# Draw filled countries
for ci, (cname, cont, geom) in enumerate(country_data):
    palette = CONTINENT_FILL_PALETTES.get(cont, CONTINENT_FILL_PALETTES['Eurasia'])
    fill_color = palette[ci % len(palette)]
    border_color = '#999999'

    polys = extract_polys(geom)
    for ext_coords, holes in polys:
        try:
            panel_ext = proj_and_panel(ext_coords)
            # Skip tiny or off-screen polygons
            if panel_ext[:, 0].max() < 0 or panel_ext[:, 0].min() > PANEL_WIDTH_INCH:
                continue
            if panel_ext[:, 1].max() < 0 or panel_ext[:, 1].min() > PANEL_HEIGHT_INCH:
                continue

            poly_patch = MplPolygon(panel_ext, closed=True,
                                     facecolor=fill_color, edgecolor=border_color,
                                     linewidth=0.15, alpha=0.85, zorder=1)
            ax_cm.add_patch(poly_patch)
        except Exception:
            continue

# Layer 2: dots + lines on top
for cont, ring_dots in continent_dots.items():
    color = CONTINENT_COLORS.get(cont, '#888888')
    for dots in ring_dots:
        if len(dots) < 2:
            continue
        closed = maybe_close(dots)
        ax_cm.plot(closed[:, 0], closed[:, 1],
                   color=color, linewidth=0.9, alpha=0.8, zorder=3)
        ax_cm.scatter(dots[:, 0], dots[:, 1],
                      s=8, c=color, zorder=4, edgecolors='white', linewidths=0.3)

# Legend
ax_cm.legend(handles=legend_handles, loc='lower left', fontsize=8,
             framealpha=0.95, edgecolor='#cccccc', fancybox=True,
             borderpad=0.8, labelspacing=0.6)

ax_cm.set_title(
    f'World Map with Dot Overlay  |  Miller  |  {total_dots} dots @ adaptive spacing',
    fontsize=11, pad=10, fontweight='bold'
)
ax_cm.set_xlabel('inch', fontsize=9)
ax_cm.set_ylabel('inch', fontsize=9)
ax_cm.set_xticks(np.arange(0, PANEL_WIDTH_INCH + 1, 10))
ax_cm.set_yticks(np.arange(0, PANEL_HEIGHT_INCH + 1, 10))
ax_cm.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('world_map_COLORED_overlay.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_COLORED_overlay.png")

# ============================================================
# Zoomed colored overlay for key regions
# ============================================================
print("Rendering zoomed colored overlay views...")

fig_zc, axes_zc = plt.subplots(1, 3, figsize=(24, 10), dpi=200)

for ax_i, (region_name, (x1, x2, y1, y2)) in enumerate(zoom_regions.items()):
    ax = axes_zc[ax_i]
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.set_aspect('equal')
    ax.set_facecolor('#cce5ff')

    # Filled countries
    for ci, (cname, cont, geom) in enumerate(country_data):
        palette = CONTINENT_FILL_PALETTES.get(cont, CONTINENT_FILL_PALETTES['Eurasia'])
        fill_color = palette[ci % len(palette)]
        polys = extract_polys(geom)
        for ext_coords, holes in polys:
            try:
                panel_ext = proj_and_panel(ext_coords)
                if panel_ext[:, 0].max() < x1 or panel_ext[:, 0].min() > x2:
                    continue
                if panel_ext[:, 1].max() < y1 or panel_ext[:, 1].min() > y2:
                    continue
                poly_patch = MplPolygon(panel_ext, closed=True,
                                         facecolor=fill_color, edgecolor='#888888',
                                         linewidth=0.2, alpha=0.85, zorder=1)
                ax.add_patch(poly_patch)
            except Exception:
                continue

    # Dots + lines
    for cont, ring_dots in continent_dots.items():
        color = CONTINENT_COLORS.get(cont, '#888888')
        for dots in ring_dots:
            if len(dots) < 2:
                continue
            closed = maybe_close(dots)
            ax.plot(closed[:, 0], closed[:, 1],
                    color=color, linewidth=1.2, alpha=0.8, zorder=2)
            ax.scatter(dots[:, 0], dots[:, 1],
                       s=20, c=color, zorder=3, edgecolors='white', linewidths=0.4)

    ax.set_title(f'{region_name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('inch', fontsize=8)
    ax.grid(True, alpha=0.1, linewidth=0.2)

plt.suptitle('Zoomed: Colored Map + Dot Overlay', fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('world_map_COLORED_zoom.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_COLORED_zoom.png")

# ============================================================
# Projector calibration version
# Corner marks + edge rulers + center crosshair
# ============================================================
print("Rendering projector calibration version...")

fig3, ax3 = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=250)
ax3.set_xlim(0, PANEL_WIDTH_INCH)
ax3.set_ylim(0, PANEL_HEIGHT_INCH)
ax3.set_aspect('equal')
ax3.set_facecolor('white')

# Panel border (thick - align to board edge)
panel_rect3 = mpatches.Rectangle(
    (0, 0), PANEL_WIDTH_INCH, PANEL_HEIGHT_INCH,
    linewidth=3, edgecolor='#e74c3c', facecolor='none'
)
ax3.add_patch(panel_rect3)

# Corner calibration marks (L-shaped, 3 inch long)
corner_len = 3
corner_lw = 2.5
corner_color = '#e74c3c'
corners = [
    (0, 0, 1, 1),                                          # bottom-left
    (PANEL_WIDTH_INCH, 0, -1, 1),                          # bottom-right
    (0, PANEL_HEIGHT_INCH, 1, -1),                         # top-left
    (PANEL_WIDTH_INCH, PANEL_HEIGHT_INCH, -1, -1),         # top-right
]
for cx, cy, dx, dy in corners:
    ax3.plot([cx, cx + dx * corner_len], [cy, cy], color=corner_color, linewidth=corner_lw, zorder=10)
    ax3.plot([cx, cx], [cy, cy + dy * corner_len], color=corner_color, linewidth=corner_lw, zorder=10)

# Center crosshair
cx, cy = PANEL_WIDTH_INCH / 2, PANEL_HEIGHT_INCH / 2
cross_size = 2
ax3.plot([cx - cross_size, cx + cross_size], [cy, cy], color='#333', linewidth=1.5, zorder=10)
ax3.plot([cx, cx], [cy - cross_size, cy + cross_size], color='#333', linewidth=1.5, zorder=10)
ax3.plot(cx, cy, 'o', color='#333', markersize=4, zorder=10)
ax3.text(cx + 0.5, cy + cross_size + 0.3, f'Center ({cx}"x{cy}")', fontsize=7, color='#333')

# Edge ruler marks every 6 inches (easy to measure with a ruler)
ruler_len = 0.8
ruler_color = '#666666'
for x in np.arange(0, PANEL_WIDTH_INCH + 1, 6):
    # Bottom edge
    ax3.plot([x, x], [0, ruler_len], color=ruler_color, linewidth=1, zorder=9)
    ax3.text(x, ruler_len + 0.2, f'{x}"', ha='center', fontsize=5, color=ruler_color)
    # Top edge
    ax3.plot([x, x], [PANEL_HEIGHT_INCH, PANEL_HEIGHT_INCH - ruler_len],
             color=ruler_color, linewidth=1, zorder=9)
for y in np.arange(0, PANEL_HEIGHT_INCH + 1, 6):
    # Left edge
    ax3.plot([0, ruler_len], [y, y], color=ruler_color, linewidth=1, zorder=9)
    ax3.text(ruler_len + 0.2, y, f'{y}"', va='center', fontsize=5, color=ruler_color)
    # Right edge
    ax3.plot([PANEL_WIDTH_INCH, PANEL_WIDTH_INCH - ruler_len], [y, y],
             color=ruler_color, linewidth=1, zorder=9)

# 4cm scale bar at bottom-left for real-world size verification
bar_x, bar_y = 4, 4
ax3.plot([bar_x, bar_x + DOT_SPACING_INCH], [bar_y, bar_y],
         color='#2c3e50', linewidth=3, zorder=10)
ax3.plot([bar_x, bar_x], [bar_y - 0.3, bar_y + 0.3],
         color='#2c3e50', linewidth=2, zorder=10)
ax3.plot([bar_x + DOT_SPACING_INCH, bar_x + DOT_SPACING_INCH],
         [bar_y - 0.3, bar_y + 0.3],
         color='#2c3e50', linewidth=2, zorder=10)
ax3.text(bar_x + DOT_SPACING_INCH / 2, bar_y + 0.6,
         f'4cm = {DOT_SPACING_INCH:.2f}"',
         ha='center', fontsize=8, color='#2c3e50', fontweight='bold')

# Also a 12-inch bar for easier ruler check
bar12_x = 4
bar12_y = 2
ax3.plot([bar12_x, bar12_x + 12], [bar12_y, bar12_y],
         color='#2c3e50', linewidth=2.5, zorder=10)
ax3.plot([bar12_x, bar12_x], [bar12_y - 0.3, bar12_y + 0.3],
         color='#2c3e50', linewidth=2, zorder=10)
ax3.plot([bar12_x + 12, bar12_x + 12], [bar12_y - 0.3, bar12_y + 0.3],
         color='#2c3e50', linewidth=2, zorder=10)
ax3.text(bar12_x + 6, bar12_y + 0.6,
         '12 inch (30.48cm)',
         ha='center', fontsize=8, color='#2c3e50', fontweight='bold')

# Draw the map (lighter, as reference)
for cont, ring_dots in continent_dots.items():
    color = CONTINENT_COLORS.get(cont, '#888888')
    for dots in ring_dots:
        if len(dots) < 2:
            continue
        closed = maybe_close(dots)
        ax3.plot(closed[:, 0], closed[:, 1],
                 color=color, linewidth=0.5, alpha=0.3, zorder=1)
        ax3.scatter(dots[:, 0], dots[:, 1],
                    s=4, c=color, zorder=2, edgecolors='none', alpha=0.5)

# Instructions
ax3.text(PANEL_WIDTH_INCH / 2, PANEL_HEIGHT_INCH - 1.5,
         'PROJECTOR CALIBRATION: Align RED border with board edges, verify scale bars with ruler',
         ha='center', fontsize=9, color='#e74c3c', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.9))

ax3.axis('off')
plt.tight_layout()
plt.savefig('world_map_CALIBRATION.png', dpi=250, bbox_inches='tight',
            facecolor='white', pad_inches=0.1)
plt.close()
print("Saved: world_map_CALIBRATION.png")

print(f"\n{'='*55}")
print(f"FINAL: Miller Projection | {total_dots} dots | {DOT_SPACING_CM}cm spacing")
print(f"Panel: {PANEL_WIDTH_INCH} x {PANEL_HEIGHT_INCH} inch")
print(f"{'='*55}")
print(f"\nProjector calibration steps:")
print(f"  1. Project world_map_CALIBRATION.png onto board")
print(f"  2. Adjust projector until RED border aligns with board edges (72x48\")")
print(f"  3. Verify with ruler: measure the 12\" scale bar = 30.48cm")
print(f"  4. Verify 4cm scale bar = actual 4cm")
print(f"  5. Switch to world_map_FINAL_clean.png and start marking dots")
