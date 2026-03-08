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

# Diablo wheel
for name in ['wheel_right_link_1', 'wheel_left_link_1']:
    path = f'resources/robots/diablo/meshes/{name}.stl'
    if os.path.exists(path):
        n, v = read_stl_info(path)
        v = v * 0.001
        print(f'Diablo {name}: {n} triangles')
        cx = (v[:,0].max()+v[:,0].min())/2
        cy = (v[:,1].max()+v[:,1].min())/2
        cz = (v[:,2].max()+v[:,2].min())/2
        r_xz = np.sqrt((v[:,0]-cx)**2 + (v[:,2]-cz)**2)
        print(f'  XZ span: {v[:,0].max()-v[:,0].min():.4f} x {v[:,2].max()-v[:,2].min():.4f}')
        print(f'  Y span: {v[:,1].max()-v[:,1].min():.4f}')
        print(f'  Radius: min={r_xz.min():.4f}, max={r_xz.max():.4f}')
        # Count unique vertices at max radius (rim)
        at_max = r_xz > 0.95 * r_xz.max()
        print(f'  Vertices near max radius: {at_max.sum()}/{len(v)}')

# Bubble wheel
for name in ['left_wheel_link_2']:
    path = f'resources/robots/bubble/meshes/{name}.stl'
    if os.path.exists(path):
        n, v = read_stl_info(path)
        v = v * 0.001
        print(f'Bubble {name}: {n} triangles')
        cx = (v[:,0].max()+v[:,0].min())/2
        cy = (v[:,1].max()+v[:,1].min())/2
        cz = (v[:,2].max()+v[:,2].min())/2
        r_xz = np.sqrt((v[:,0]-cx)**2 + (v[:,2]-cz)**2)
        print(f'  XZ span: {v[:,0].max()-v[:,0].min():.4f} x {v[:,2].max()-v[:,2].min():.4f}')
        print(f'  Y span: {v[:,1].max()-v[:,1].min():.4f}')
        print(f'  Radius: min={r_xz.min():.4f}, max={r_xz.max():.4f}')
        at_max = r_xz > 0.95 * r_xz.max()
        print(f'  Vertices near max radius: {at_max.sum()}/{len(v)}')
