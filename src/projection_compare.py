"""
Compare intermediate projections between Mercator and Equirectangular.
Goal: Greenland not too big (Mercator) nor too small (Equirect), Africa/Australia fair size.
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
PANEL_WIDTH_INCH = 76
PANEL_HEIGHT_INCH = 48
MARGIN_INCH = 2.0
DOT_SPACING_CM = 4.0
INCH_TO_CM = 2.54
DOT_SPACING_INCH = DOT_SPACING_CM / INCH_TO_CM
DRAW_WIDTH = PANEL_WIDTH_INCH - 2 * MARGIN_INCH
DRAW_HEIGHT = PANEL_HEIGHT_INCH - 2 * MARGIN_INCH

MERGE_EURASIA = True  # True: merge Europe+Asia into Eurasia

CONTINENT_COLORS = {
    'Eurasia':       '#e74c3c',
    'Asia':          '#e74c3c',
    'Europe':        '#3498db',
    'Africa':        '#f39c12',
    'North America': '#2ecc71',
    'South America': '#9b59b6',
    'Oceania':       '#1abc9c',
    'Antarctica':    '#95a5a6',
}

# ============================================================
# Country -> Continent (reuse from v2)
# ============================================================
CONTINENT_MAP = {}
_ASIA = [
    'AFG','ARM','AZE','BHR','BGD','BTN','BRN','KHM','CHN','CYP','GEO','IND',
    'IDN','IRN','IRQ','ISR','JPN','JOR','KAZ','KWT','KGZ','LAO','LBN','MYS',
    'MDV','MNG','MMR','NPL','OMN','PAK','PSE','PHL','QAT','SAU','SGP','KOR',
    'PRK','LKA','SYR','TWN','TJK','THA','TLS','TKM','ARE','UZB','VNM',
    'YEM','IOT','CCK','CXR','HKG','MAC','XAD','XKS','BIO','XPI','BJN',
    'CSG','CNM','ESB','WSB','SAH','SER','SCR','XNC','USG','KAS','KAB',
    'RUS','TUR',
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
for codes, continent in [
    (_ASIA, 'Asia'), (_EUROPE, 'Europe'), (_AFRICA, 'Africa'),
    (_NORTH_AMERICA, 'North America'), (_SOUTH_AMERICA, 'South America'),
    (_OCEANIA, 'Oceania'), (['ATA'], 'Antarctica'),
]:
    for c in codes:
        CONTINENT_MAP[c] = continent

def guess_continent(lon, lat):
    if lat < -60: return 'Antarctica'
    if lat > 60 and lon < -10: return 'North America'
    if lon < -30: return 'North America' if lat > 15 else 'South America'
    if lon < 60: return 'Europe' if lat > 35 else 'Africa'
    if lon < 150: return 'Asia'
    return 'Oceania'

# ============================================================
# Projection functions
# ============================================================
def proj_equirectangular(lon, lat):
    return lon.copy(), lat.copy()

def proj_mercator(lon, lat):
    lat_c = np.clip(lat, -80, 80)
    lat_r = np.radians(lat_c)
    y = np.degrees(np.log(np.tan(np.pi/4 + lat_r/2)))
    return lon.copy(), y

def proj_miller(lon, lat):
    """Miller Cylindrical: like Mercator but 80% of the vertical stretch.
    Greenland smaller than Mercator, bigger than Equirectangular."""
    lat_r = np.radians(np.clip(lat, -85, 85))
    y = np.degrees(1.25 * np.log(np.tan(np.pi/4 + 0.4 * lat_r)))
    return lon.copy(), y

def proj_compact_miller(lon, lat):
    """Compact Miller: even less polar stretch than Miller.
    A nice middle ground."""
    lat_r = np.radians(np.clip(lat, -85, 85))
    y = np.degrees(1.1 * np.log(np.tan(np.pi/4 + 0.35 * lat_r)))
    return lon.copy(), y

def proj_kavrayskiy(lon, lat):
    """Kavrayskiy VII: pseudocylindrical, good compromise.
    Used in many Russian atlases."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.degrees(lon_r * np.sqrt(1.0/3 - (lat_r / np.pi)**2))
    y = lat.copy()
    return x, y

def proj_natural_earth(lon, lat):
    """Natural Earth projection: designed by Tom Patterson for cartography.
    Very balanced, pleasant to look at."""
    # Polynomial approximation
    lat_r = np.radians(lat)
    lat2 = lat_r * lat_r
    lat4 = lat2 * lat2
    lat6 = lat2 * lat4

    x_coeff = 0.8707 - 0.131979 * lat2 + 0.003971 * lat4 - 0.001529 * lat6
    y_coeff = lat_r * (1.007226 + 0.015085 * lat2 - 0.044475 * lat4 + 0.028874 * lat6 - 0.005916 * lat2*lat4)

    x = np.degrees(np.radians(lon) * x_coeff)
    y = np.degrees(y_coeff)
    return x, y

def proj_winkel_tripel(lon, lat):
    """Winkel Tripel: used by National Geographic. Very balanced."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    # Aitoff component
    alpha = np.arccos(np.clip(np.cos(lat_r) * np.cos(lon_r / 2), -1, 1))
    sinc_alpha = np.where(np.abs(alpha) < 1e-10, 1.0, np.sin(alpha) / alpha)

    x_aitoff = 2 * np.cos(lat_r) * np.sin(lon_r / 2) / sinc_alpha
    y_aitoff = np.sin(lat_r) / sinc_alpha

    # Equirectangular component with standard parallel at ~50.47 deg
    cos_phi1 = np.cos(np.radians(50.4667))
    x_equi = lon_r * cos_phi1
    y_equi = lat_r

    x = np.degrees(0.5 * (x_aitoff + x_equi))
    y = np.degrees(0.5 * (y_aitoff + y_equi))
    return x, y

# All projections to compare
PROJECTIONS = {
    'Equirectangular':  (proj_equirectangular, 'Greenland too small\nAfrica/Australia fair'),
    'Compact Miller':   (proj_compact_miller,  'Mild stretch\nGood middle ground'),
    'Miller':           (proj_miller,          'Like Mercator lite\n80% of the stretch'),
    'Natural Earth':    (proj_natural_earth,   'Cartographer favorite\nVery balanced'),
    'Kavrayskiy VII':   (proj_kavrayskiy,      'Russian atlas style\nClean, balanced'),
    'Winkel Tripel':    (proj_winkel_tripel,   'National Geographic\nMinimal distortion'),
    'Mercator':         (proj_mercator,        'Greenland too big\nAfrica/Australia small'),
}

# ============================================================
# Load data
# ============================================================
print("Loading...")
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

# Transcontinental countries: split by longitude
# Russia: west of 60E = Europe, east of 60E = Asia
# Turkey: west of 29E = Europe, east of 29E = Asia
# Egypt: west of 33E (Sinai) = Africa, east = Asia (but keep all in Africa for simplicity)
SPLIT_COUNTRIES = {
    'RUS': (60.0, 'Europe', 'Asia'),   # Ural Mountains ~60E
    'TUR': (29.0, 'Europe', 'Asia'),   # Bosphorus ~29E
}

def split_geom_by_lon(geom, split_lon):
    """Split a geometry into west and east parts at a given longitude."""
    west_box = box(-180, -90, split_lon, 90)
    east_box = box(split_lon, -90, 180, 90)
    west = geom.intersection(west_box)
    east = geom.intersection(east_box)
    return west, east

# Group by continent
continent_geoms = {}
for feature in data['features']:
    props = feature['properties']
    iso3 = props.get('ISO3166-1-Alpha-3', '')
    name = props.get('name', '')

    geom = shape(feature['geometry'])
    if not geom.is_valid:
        geom = geom.buffer(0)

    # Handle transcontinental countries
    if iso3 in SPLIT_COUNTRIES and not MERGE_EURASIA:
        split_lon, west_cont, east_cont = SPLIT_COUNTRIES[iso3]
        west_geom, east_geom = split_geom_by_lon(geom, split_lon)
        for part, cont in [(west_geom, west_cont), (east_geom, east_cont)]:
            if not part.is_empty:
                if cont not in continent_geoms:
                    continent_geoms[cont] = []
                continent_geoms[cont].append(part)
        print(f"  Split {name} ({iso3}) at {split_lon}E -> {west_cont}/{east_cont}")
        continue

    continent = CONTINENT_MAP.get(iso3)
    if not continent:
        continent = guess_continent(geom.centroid.x, geom.centroid.y)
    if continent == 'Antarctica':
        continue
    if continent not in continent_geoms:
        continent_geoms[continent] = []
    continent_geoms[continent].append(geom)

# Merge Europe + Asia -> Eurasia if enabled
if MERGE_EURASIA:
    eurasia_geoms = []
    for key in ['Europe', 'Asia']:
        if key in continent_geoms:
            eurasia_geoms.extend(continent_geoms.pop(key))
    if eurasia_geoms:
        continent_geoms['Eurasia'] = eurasia_geoms
        print(f"  Merged Europe + Asia -> Eurasia ({len(eurasia_geoms)} geometries)")

# Merge per continent
continent_rings_raw = {}
for continent, geoms in continent_geoms.items():
    merged = unary_union(geoms)
    rings = extract_rings(merged)
    # Fix antimeridian: shift negative-lon polygons that belong to the eastern
    # hemisphere (e.g. Russia's Chukotka at -180..-169) to positive side (+180..+191)
    # so the continent stays on one side of the map.
    fixed_rings = []
    for ring in rings:
        lon_mean = ring[:, 0].mean()
        lon_min = ring[:, 0].min()
        lon_max = ring[:, 0].max()
        lon_span = lon_max - lon_min

        if lon_span > 200:
            # This ring wraps around the antimeridian - split into two
            # by shifting negative coords to positive
            fixed = ring.copy()
            mask = fixed[:, 0] < 0
            fixed[mask, 0] += 360
            fixed_rings.append(fixed)
            print(f"  Fixed antimeridian wrap in {continent} ring (span {lon_span:.0f} deg)")
        elif lon_mean < -100 and continent in ('Eurasia', 'Asia'):
            # Small polygon on wrong side (e.g. Chukotka islands at -172)
            fixed = ring.copy()
            fixed[:, 0] += 360
            fixed_rings.append(fixed)
        else:
            fixed_rings.append(ring)
    continent_rings_raw[continent] = fixed_rings

print(f"Loaded {sum(len(r) for r in continent_rings_raw.values())} rings across {len(continent_rings_raw)} continents")

# ============================================================
# Dot placement
# ============================================================
def place_dots(ring_coords, spacing):
    n = len(ring_coords)
    if n < 2: return np.array([])
    diffs = np.diff(ring_coords, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    total = seg_lengths.sum()
    if total < spacing * 0.5: return np.array([])
    num = max(3, int(round(total / spacing)))
    actual = total / num
    cum = np.zeros(n)
    cum[1:] = np.cumsum(seg_lengths)
    dots = []
    for i in range(num):
        target = i * actual
        idx = np.searchsorted(cum, target, side='right') - 1
        idx = max(0, min(idx, n-2))
        sl = seg_lengths[idx]
        t = np.clip((target - cum[idx]) / sl, 0, 1) if sl > 0 else 0
        dots.append(ring_coords[idx] + t * (ring_coords[idx+1] - ring_coords[idx]))
    return np.array(dots)

MIN_PERIM = DOT_SPACING_INCH * 3

# ============================================================
# Generate comparison: 7 projections in a grid
# ============================================================
print("Generating comparison...")

n_proj = len(PROJECTIONS)
fig, axes = plt.subplots(2, 4, figsize=(28, 14), dpi=150)
axes_flat = axes.flatten()

proj_results = {}

for ax_i, (pname, (pfn, pdesc)) in enumerate(PROJECTIONS.items()):
    ax = axes_flat[ax_i]

    # Project all rings
    all_proj = []
    continent_proj = {}
    for cont, rings in continent_rings_raw.items():
        proj_rings = []
        for ring in rings:
            lon, lat = ring[:, 0], ring[:, 1]
            x, y = pfn(lon, lat)
            proj_rings.append(np.column_stack([x, y]))
            all_proj.append(proj_rings[-1])
        continent_proj[cont] = proj_rings

    # Scale to panel
    pts = np.vstack(all_proj)
    xn, yn = pts.min(axis=0)
    xx, yx = pts.max(axis=0)
    xr, yr = xx - xn, yx - yn
    sc = min(DRAW_WIDTH / xr, DRAW_HEIGHT / yr)
    ox = MARGIN_INCH + (DRAW_WIDTH - xr * sc) / 2
    oy = MARGIN_INCH + (DRAW_HEIGHT - yr * sc) / 2

    ax.set_xlim(0, PANEL_WIDTH_INCH)
    ax.set_ylim(0, PANEL_HEIGHT_INCH)
    ax.set_aspect('equal')
    ax.set_facecolor('#faf8f2')

    total_n = 0
    for cont, proj_rings in continent_proj.items():
        color = CONTINENT_COLORS.get(cont, '#888')
        for ring in proj_rings:
            pr = np.zeros_like(ring)
            pr[:, 0] = (ring[:, 0] - xn) * sc + ox
            pr[:, 1] = (ring[:, 1] - yn) * sc + oy
            diffs = np.diff(pr, axis=0)
            perim = np.sqrt((diffs**2).sum(axis=1)).sum()
            if perim < MIN_PERIM:
                continue
            dots = place_dots(pr, DOT_SPACING_INCH)
            if len(dots) > 0:
                total_n += len(dots)
                closed = np.vstack([dots, dots[0:1]])
                ax.plot(closed[:, 0], closed[:, 1],
                        color=color, linewidth=0.6, alpha=0.7, zorder=1)
                ax.scatter(dots[:, 0], dots[:, 1],
                           s=3, c=color, zorder=2, edgecolors='white', linewidths=0.15)

    proj_results[pname] = total_n

    ax.set_title(f'{pname}\n{pdesc}\n{total_n} dots', fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    rect = mpatches.Rectangle((0.5, 0.5), PANEL_WIDTH_INCH-1, PANEL_HEIGHT_INCH-1,
                               linewidth=0.5, edgecolor='#aaa', facecolor='none')
    ax.add_patch(rect)

# Hide the 8th subplot
axes_flat[7].axis('off')

# Summary text in the empty subplot
summary = "Projection Summary\n(dots @ 4cm spacing)\n\n"
for pname, ndots in proj_results.items():
    summary += f"{pname}: {ndots}\n"
axes_flat[7].text(0.5, 0.5, summary, transform=axes_flat[7].transAxes,
                  fontsize=11, ha='center', va='center', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle(
    f'Projection Comparison  |  Panel {PANEL_WIDTH_INCH}x{PANEL_HEIGHT_INCH}"  |  '
    f'{DOT_SPACING_CM}cm dot spacing  |  No Antarctica',
    fontsize=13, fontweight='bold', y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('world_map_proj_compare.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_proj_compare.png")

# ============================================================
# Also render the top 3 candidates larger for detail comparison
# ============================================================
print("Rendering top 3 candidates in detail...")

top3 = ['Compact Miller', 'Miller', 'Natural Earth']

fig2, axes2 = plt.subplots(1, 3, figsize=(27, 10), dpi=180)

for ax_i, pname in enumerate(top3):
    pfn, pdesc = PROJECTIONS[pname]
    ax = axes2[ax_i]

    all_proj = []
    continent_proj = {}
    for cont, rings in continent_rings_raw.items():
        proj_rings = []
        for ring in rings:
            lon, lat = ring[:, 0], ring[:, 1]
            x, y = pfn(lon, lat)
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

    ax.set_xlim(0, PANEL_WIDTH_INCH)
    ax.set_ylim(0, PANEL_HEIGHT_INCH)
    ax.set_aspect('equal')
    ax.set_facecolor('#faf8f2')

    total_n = 0
    legend_data = {}
    for cont, proj_rings in continent_proj.items():
        color = CONTINENT_COLORS.get(cont, '#888')
        cont_n = 0
        for ring in proj_rings:
            pr = np.zeros_like(ring)
            pr[:, 0] = (ring[:, 0] - xn) * sc + ox
            pr[:, 1] = (ring[:, 1] - yn) * sc + oy
            diffs = np.diff(pr, axis=0)
            perim = np.sqrt((diffs**2).sum(axis=1)).sum()
            if perim < MIN_PERIM:
                continue
            dots = place_dots(pr, DOT_SPACING_INCH)
            if len(dots) > 0:
                total_n += len(dots)
                cont_n += len(dots)
                closed = np.vstack([dots, dots[0:1]])
                ax.plot(closed[:, 0], closed[:, 1],
                        color=color, linewidth=0.8, alpha=0.7, zorder=1)
                ax.scatter(dots[:, 0], dots[:, 1],
                           s=6, c=color, zorder=2, edgecolors='white', linewidths=0.2)
        if cont_n > 0:
            legend_data[cont] = cont_n

    # Legend
    handles = []
    for cont in sorted(legend_data.keys()):
        color = CONTINENT_COLORS.get(cont, '#888')
        handles.append(Line2D([0], [0], marker='o', color=color, linewidth=1,
                              markersize=5, markerfacecolor=color, markeredgecolor='white',
                              markeredgewidth=0.3, label=f'{cont} ({legend_data[cont]})'))
    ax.legend(handles=handles, loc='lower left', fontsize=7, framealpha=0.9)

    ax.set_title(f'{pname}\n{pdesc}\nTotal: {total_n} dots', fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(0, PANEL_WIDTH_INCH+1, 10))
    ax.set_yticks(np.arange(0, PANEL_HEIGHT_INCH+1, 10))
    ax.grid(True, alpha=0.1, linewidth=0.3)

plt.suptitle('Top 3 Recommended Projections (between Mercator and Equirectangular)',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('world_map_proj_top3.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: world_map_proj_top3.png")

print("\nDone!")
for pname, ndots in proj_results.items():
    print(f"  {pname:20s}: {ndots} dots")
