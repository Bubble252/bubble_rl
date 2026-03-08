import struct, os, numpy as np

def read_stl_info(path):
    with open(path, 'rb') as f:
        header = f.read(80)
        num_tri = struct.unpack('<I', f.read(4))[0]
        verts = []
        for _ in range(num_tri):
            data = struct.unpack('<12fH', f.read(50))
            verts.append(data[3:6])
            verts.append(data[6:9])
            verts.append(data[9:12])
    return num_tri, np.array(verts)

def analyze_roundness(name, path, scale=0.001):
    n, v = read_stl_info(path)
    v = v * scale
    
    cx = (v[:,0].max()+v[:,0].min())/2
    cz = (v[:,2].max()+v[:,2].min())/2
    r_xz = np.sqrt((v[:,0]-cx)**2 + (v[:,2]-cz)**2)
    
    # Get outer rim vertices (top 10% radius)
    r_max = r_xz.max()
    outer = r_xz > 0.90 * r_max
    outer_v = v[outer]
    outer_r = r_xz[outer]
    
    print(f'\n=== {name} ===')
    print(f'  Total triangles: {n}')
    print(f'  Max radius: {r_max:.6f}')
    print(f'  Outer rim vertices (>90% max_r): {outer.sum()}')
    print(f'  Outer radius: min={outer_r.min():.6f}, max={outer_r.max():.6f}, std={outer_r.std():.6f}')
    print(f'  Roundness (std/mean): {outer_r.std()/outer_r.mean()*100:.2f}%')
    
    # Compute angles around Y axis for outer vertices
    angles = np.arctan2(outer_v[:,2]-cz, outer_v[:,0]-cx)
    # Bin into 36 sectors (10 degrees each) and check coverage
    n_sectors = 36
    sector_counts = np.zeros(n_sectors)
    for a in angles:
        sector = int((a + np.pi) / (2*np.pi) * n_sectors) % n_sectors
        sector_counts[sector] += 1
    empty_sectors = (sector_counts == 0).sum()
    print(f'  Angular coverage: {n_sectors - empty_sectors}/{n_sectors} sectors filled')
    print(f'  Sector vertex counts: min={sector_counts.min():.0f}, max={sector_counts.max():.0f}')
    
    # Compute convex hull radius variation
    # For each angle sector, find max radius (this is what convex hull would use)
    hull_r = []
    for i in range(n_sectors):
        mask = ((angles + np.pi) / (2*np.pi) * n_sectors).astype(int) % n_sectors == i
        if mask.any():
            hull_r.append(r_xz[outer][mask].max())
    hull_r = np.array(hull_r)
    print(f'  Convex hull radius per sector: min={hull_r.min():.6f}, max={hull_r.max():.6f}, std={hull_r.std():.6f}')
    print(f'  Hull roundness deviation: {(hull_r.max()-hull_r.min())/hull_r.mean()*100:.2f}%')

# Diablo
analyze_roundness('Diablo wheel', 'resources/robots/diablo/meshes/wheel_right_link_1.stl')

# Bubble
analyze_roundness('Bubble wheel', 'resources/robots/bubble/meshes/left_wheel_link_2.stl')
