"""
Post-process checkpoint: enforce minimum pin spacing.
Board: 72x48 inches. Coordinates in inches.
Minimum spacing: 2cm = 0.787 inches.

Strategy:
1. Load checkpoint (all pin coordinates)
2. Build global KD-tree of ALL pins across all rings
3. Find all pairs closer than MIN_SPACING
4. For each too-close pair, try to:
   a. Slide one pin along its coastline to increase spacing
   b. If can't fix by sliding, remove the pin with lower IoU contribution
5. After spacing enforcement, trim to TARGET dots if needed
6. Save new checkpoint + render
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
import json

# ============================================================
# Config
# ============================================================
PANEL_W, PANEL_H = 72, 48  # inches
MIN_SPACING_CM = 2.0
MIN_SPACING_INCH = MIN_SPACING_CM / 2.54  # 0.787 inches
TARGET = 1100

CONTINENT_COLORS = {
    'Eurasia': '#E53935', 'North America': '#43A047',
    'Africa': '#FB8C00', 'South America': '#8E24AA',
    'Oceania': '#00ACC1'
}

# ============================================================
# Load checkpoint
# ============================================================
print("Loading checkpoint...")
data = np.load('world_map_checkpoint.npz', allow_pickle=True)
n_rings = int(data['n_rings'])
iou_orig = float(data['iou'])

rings = []
meta = []
for i in range(n_rings):
    rings.append(data[f'ring_{i}'].copy())
    meta.append({'continent': str(data[f'cont_{i}'])})

total_before = sum(len(r) for r in rings)
print(f"Loaded: {n_rings} rings, {total_before} dots, IoU={iou_orig:.4f}")

# ============================================================
# Step 1: Analyze current spacing violations
# ============================================================
print(f"\nMinimum spacing: {MIN_SPACING_CM}cm = {MIN_SPACING_INCH:.3f} inches")

# Build flat array of all points with (ring_idx, point_idx) mapping
all_pts = []
pt_map = []  # (ring_idx, point_idx)
for ri, ring in enumerate(rings):
    for pi, pt in enumerate(ring):
        all_pts.append(pt)
        pt_map.append((ri, pi))
all_pts = np.array(all_pts)

tree = cKDTree(all_pts)
raw_pairs = tree.query_pairs(r=MIN_SPACING_INCH)

# Filter: only keep pairs that are NOT adjacent in the same ring
# Adjacent = same ring AND (|pi - pj| == 1 or wrap-around)
pairs = set()
for i, j in raw_pairs:
    ri_i, pi_i = pt_map[i]
    ri_j, pi_j = pt_map[j]
    if ri_i == ri_j:
        # Same ring - check if adjacent
        n = len(rings[ri_i])
        diff = abs(pi_i - pi_j)
        if diff == 1 or diff == n - 1:
            continue  # adjacent in same ring, skip
    pairs.add((i, j))
print(f"Spacing violations (<{MIN_SPACING_CM}cm): {len(pairs)} pairs (excluded same-ring adjacent)")

# Show distribution of distances
if pairs:
    dists = []
    for i, j in pairs:
        d = np.sqrt(((all_pts[i] - all_pts[j])**2).sum())
        dists.append(d)
    dists = np.array(dists)
    print(f"  Min distance: {dists.min()*2.54:.2f}cm")
    print(f"  Mean distance: {dists.mean()*2.54:.2f}cm")
    print(f"  Violations <1cm: {(dists < 1/2.54).sum()}")
    print(f"  Violations 1-2cm: {((dists >= 1/2.54) & (dists < 2/2.54)).sum()}")

# ============================================================
# Step 2: Remove dots that are too close
# ============================================================
print(f"\nStep 2: Enforcing spacing by removing violations...")

# Score each point by how many violations it has
violation_count = np.zeros(len(all_pts), dtype=int)
for i, j in pairs:
    violation_count[i] += 1
    violation_count[j] += 1

# Also compute each point's "importance" = distance to neighbors in same ring
# Points on sharp corners are more important (large angle change)
importance = np.zeros(len(all_pts))
flat_idx = 0
for ri, ring in enumerate(rings):
    n = len(ring)
    for pi in range(n):
        if n < 3:
            importance[flat_idx] = 1.0
        else:
            p_prev = ring[(pi - 1) % n]
            p_curr = ring[pi]
            p_next = ring[(pi + 1) % n]
            # Angle at this point - sharper angles = more important
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            l1 = np.sqrt((v1**2).sum())
            l2 = np.sqrt((v2**2).sum())
            if l1 > 0 and l2 > 0:
                cos_angle = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
                angle = np.arccos(cos_angle)
                importance[flat_idx] = angle  # 0 = straight line, pi = sharp corner
            else:
                importance[flat_idx] = 0
        flat_idx += 1

# Iteratively remove violations in multiple passes until clean
removed = set()
pass_num = 0
while True:
    pass_num += 1
    # Rebuild tree without removed points
    active = [i for i in range(len(all_pts)) if i not in removed]
    if not active:
        break
    active_pts = all_pts[active]
    active_tree = cKDTree(active_pts)
    active_pairs = active_tree.query_pairs(r=MIN_SPACING_INCH)

    # Filter adjacent
    real_violations = []
    for ai, aj in active_pairs:
        i, j = active[ai], active[aj]
        ri_i, pi_i = pt_map[i]
        ri_j, pi_j = pt_map[j]
        if ri_i == ri_j:
            n = len(rings[ri_i])
            diff = abs(pi_i - pi_j)
            if diff == 1 or diff == n - 1:
                continue
        real_violations.append((i, j))

    if not real_violations:
        break

    # Sort by importance
    real_violations.sort(key=lambda p: min(importance[p[0]], importance[p[1]]))

    new_removed = 0
    for i, j in real_violations:
        if i in removed or j in removed:
            continue
        if importance[i] <= importance[j]:
            removed.add(i)
        else:
            removed.add(j)
        new_removed += 1

    print(f"  Pass {pass_num}: removed {new_removed}, total removed {len(removed)}, violations left {len(real_violations)}")
    if new_removed == 0:
        break

print(f"  Total removed: {len(removed)} dots")

# Rebuild rings without removed points
removed_by_ring = {}
for flat_i in removed:
    ri, pi = pt_map[flat_i]
    if ri not in removed_by_ring:
        removed_by_ring[ri] = set()
    removed_by_ring[ri].add(pi)

new_rings = []
for ri, ring in enumerate(rings):
    if ri in removed_by_ring:
        keep = [pi for pi in range(len(ring)) if pi not in removed_by_ring[ri]]
        if len(keep) >= 3:  # need at least 3 points for a valid ring
            new_rings.append(ring[keep])
        else:
            new_rings.append(ring)  # keep as is if too few remain
    else:
        new_rings.append(ring.copy())

total_after_spacing = sum(len(r) for r in new_rings)
print(f"  Dots after spacing: {total_after_spacing}")

# ============================================================
# Step 3: Trim to TARGET if needed
# ============================================================
if total_after_spacing > TARGET:
    print(f"\nStep 3: Trimming {total_after_spacing} -> {TARGET}...")
    # Remove least important points globally until we hit target
    # Rebuild importance for remaining points
    all_pts2 = []
    pt_map2 = []
    imp2 = []
    flat_idx = 0
    for ri, ring in enumerate(new_rings):
        n = len(ring)
        for pi in range(n):
            all_pts2.append(ring[pi])
            pt_map2.append((ri, pi))
            if n < 4:
                imp2.append(float('inf'))  # don't remove from tiny rings
            else:
                p_prev = ring[(pi - 1) % n]
                p_curr = ring[pi]
                p_next = ring[(pi + 1) % n]
                v1 = p_prev - p_curr
                v2 = p_next - p_curr
                l1 = np.sqrt((v1**2).sum())
                l2 = np.sqrt((v2**2).sum())
                if l1 > 0 and l2 > 0:
                    cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
                    imp2.append(np.arccos(cos_a))
                else:
                    imp2.append(0)

    imp2 = np.array(imp2)
    # Sort by importance (ascending = least important first)
    order = np.argsort(imp2)

    to_remove = total_after_spacing - TARGET
    trim_removed = set()
    for idx in order:
        if len(trim_removed) >= to_remove:
            break
        ri, pi = pt_map2[idx]
        if len(new_rings[ri]) - sum(1 for x in trim_removed if pt_map2[x][0] == ri) <= 3:
            continue  # keep at least 3 points per ring
        trim_removed.add(idx)

    # Rebuild
    trim_by_ring = {}
    for idx in trim_removed:
        ri, pi = pt_map2[idx]
        if ri not in trim_by_ring:
            trim_by_ring[ri] = set()
        trim_by_ring[ri].add(pi)

    for ri in trim_by_ring:
        keep = [pi for pi in range(len(new_rings[ri])) if pi not in trim_by_ring[ri]]
        new_rings[ri] = new_rings[ri][keep]

    total_after_trim = sum(len(r) for r in new_rings)
    print(f"  Dots after trim: {total_after_trim}")
else:
    print(f"\nAlready at or below target ({total_after_spacing} <= {TARGET})")

# ============================================================
# Step 4: Verify spacing
# ============================================================
print("\nStep 4: Verifying spacing...")
all_final = []
pt_map_final = []
for ri, ring in enumerate(new_rings):
    for pi, pt in enumerate(ring):
        all_final.append(pt)
        pt_map_final.append((ri, pi))
all_final = np.array(all_final)
tree_final = cKDTree(all_final)
raw_remaining = tree_final.query_pairs(r=MIN_SPACING_INCH)
# Filter adjacent
remaining_violations = set()
for i, j in raw_remaining:
    ri_i, pi_i = pt_map_final[i]
    ri_j, pi_j = pt_map_final[j]
    if ri_i == ri_j:
        n = len(new_rings[ri_i])
        diff = abs(pi_i - pi_j)
        if diff == 1 or diff == n - 1:
            continue
    remaining_violations.add((i, j))
print(f"  Remaining violations (non-adjacent): {len(remaining_violations)}")
if remaining_violations:
    vdists = []
    for i, j in remaining_violations:
        vdists.append(np.sqrt(((all_final[i] - all_final[j])**2).sum()))
    vdists = np.array(vdists)
    print(f"  Min distance: {vdists.min()*2.54:.2f}cm")

total_final = len(all_final)

# ============================================================
# Step 5: Render
# ============================================================
print(f"\nStep 5: Rendering ({total_final} dots)...")

fw, fh = 14, 9.33
fig, ax = plt.subplots(1, 1, figsize=(fw, fh), dpi=200)
ax.set_xlim(0, PANEL_W)
ax.set_ylim(PANEL_H, 0)
ax.set_aspect('equal')
ax.set_facecolor('#f5f0e8')

cg = {}
for ri, d in enumerate(new_rings):
    co = meta[ri]['continent']
    if co not in cg: cg[co] = []
    cg[co].append(d)

continent_dots = {}
for co, rl in cg.items():
    cl = CONTINENT_COLORS.get(co, '#888')
    continent_dots[co] = rl
    for d in rl:
        if len(d) < 2: continue
        cs = np.vstack([d, d[0:1]])
        ax.plot(cs[:,0], cs[:,1], color=cl, linewidth=0.8, alpha=0.7, zorder=2)
        ax.scatter(d[:,0], d[:,1], s=24, c=cl, zorder=3, edgecolors='white', linewidths=0.4)

ax.set_title(f'World Map  |  {total_final} dots  |  Min spacing {MIN_SPACING_CM}cm  |  Big Pins',
             fontsize=10, fontweight='bold', pad=8)
ax.set_xticks(np.arange(0, PANEL_W+1, 10))
ax.set_yticks(np.arange(0, PANEL_H+1, 10))
ax.tick_params(labelsize=6)

handles = []
for co in sorted(continent_dots.keys()):
    cl = CONTINENT_COLORS.get(co, '#888')
    n = sum(len(d) for d in continent_dots[co])
    if n > 0:
        handles.append(Line2D([0],[0], marker='o', color=cl, linewidth=1, markersize=12,
                              markerfacecolor=cl, markeredgecolor='white', markeredgewidth=0.5,
                              label=f'{co} ({n})'))
ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.95)
plt.tight_layout()

out = f'../results/final_{total_final}dots_spaced_{MIN_SPACING_CM}cm.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")

# Save new checkpoint
np.savez('world_map_checkpoint_spaced.npz',
         n_rings=len(new_rings),
         iou=0,  # not recomputed
         **{f'ring_{i}': new_rings[i] for i in range(len(new_rings))},
         **{f'cont_{i}': meta[i]['continent'] for i in range(len(new_rings))})
print("Saved: world_map_checkpoint_spaced.npz")

# Save coordinates as JSON
pin_data = []
for ri, d in enumerate(new_rings):
    pin_data.append({'continent': meta[ri]['continent'], 'points': d.tolist()})
with open(f'../results/pin_coordinates_spaced_{MIN_SPACING_CM}cm.json', 'w') as f:
    json.dump(pin_data, f)
print(f"Saved: pin_coordinates_spaced_{MIN_SPACING_CM}cm.json")

print(f"\n{'='*50}")
print(f"DONE: {total_before} -> {total_final} dots")
print(f"Spacing violations: {len(pairs)} -> {len(remaining_violations)}")
print(f"{'='*50}")
