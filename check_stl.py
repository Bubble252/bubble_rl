import numpy as np
import struct, os

def read_stl_binary(path):
    with open(path, 'rb') as f:
        header = f.read(80)
        num_tri = struct.unpack('<I', f.read(4))[0]
        verts = []
        for _ in range(num_tri):
            data = struct.unpack('<12fH', f.read(50))
            verts.append(data[3:6])
            verts.append(data[6:9])
            verts.append(data[9:12])
    return np.array(verts)

for name in ['left_wheel_link_2', 'right_wheel_link_1', 'left_idler_wheel_link_2', 'right_idler_wheel_link_1']:
    path = f'resources/robots/bubble/meshes/{name}.stl'
    if os.path.exists(path):
        v = read_stl_binary(path) * 0.001  # scale same as URDF
        print(f'\n=== {name} ===')
        print(f'  Vertices: {len(v)}')
        print(f'  X range: [{v[:,0].min():.6f}, {v[:,0].max():.6f}]  span={v[:,0].max()-v[:,0].min():.6f}')
        print(f'  Y range: [{v[:,1].min():.6f}, {v[:,1].max():.6f}]  span={v[:,1].max()-v[:,1].min():.6f}')
        print(f'  Z range: [{v[:,2].min():.6f}, {v[:,2].max():.6f}]  span={v[:,2].max()-v[:,2].min():.6f}')
        cx = (v[:,0].max()+v[:,0].min())/2
        cy = (v[:,1].max()+v[:,1].min())/2
        cz = (v[:,2].max()+v[:,2].min())/2
        print(f'  Center: ({cx:.6f}, {cy:.6f}, {cz:.6f})')
        r_xz = np.sqrt((v[:,0]-cx)**2 + (v[:,2]-cz)**2)
        print(f'  Radius (XZ from center): min={r_xz.min():.6f}, max={r_xz.max():.6f}, mean={r_xz.mean():.6f}')
    else:
        print(f'{name}: NOT FOUND')
