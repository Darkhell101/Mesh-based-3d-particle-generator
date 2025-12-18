import open3d as o3d
import numpy as np
import pyvista as pv


def read_mesh_and_check(ply_path):
    """读取网格并检查/修复水密性"""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if not mesh.is_watertight:
        print("警告：网格不封闭，体积计算可能不准，正在尝试修复...")
        mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def build_ray_casting_scene(mesh):
    """构建用于点-体判断的射线场景"""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def generate_hcp_lattice(r, bbox_min, bbox_max):
    """生成六方最密堆积（HCP）候选点（当前性能可接受）"""
    a = 2 * r
    h = a * np.sqrt(6) / 3
    pts = []
    z = bbox_min[2]
    shift_layer = False

    while z <= bbox_max[2]:
        y = bbox_min[1]
        while y <= bbox_max[1]:
            x_start = bbox_min[0] + (a / 2 if shift_layer else 0)
            x = x_start
            while x <= bbox_max[0]:
                pts.append([x, y, z])
                x += a
            y += a * np.sqrt(3) / 2
        shift_layer = not shift_layer
        z += h

    return np.array(pts, dtype=np.float32)


def generate_scc_lattice(r, bbox_min, bbox_max):
    """生成简单立方堆积（SCC）候选点（已向量化）"""
    step = 2 * r
    x = np.arange(bbox_min[0], bbox_max[0] + step / 2, step)
    y = np.arange(bbox_min[1], bbox_max[1] + step / 2, step)
    z = np.arange(bbox_min[2], bbox_max[2] + step / 2, step)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def generate_fcc_lattice(r, bbox_min, bbox_max):
    """向量化生成面心立方堆积（FCC）候选点"""
    a = 2 * np.sqrt(2) * r
    # 显式使用 float32
    x0 = np.arange(bbox_min[0], bbox_max[0] + a / 2, a, dtype=np.float32)
    y0 = np.arange(bbox_min[1], bbox_max[1] + a / 2, a, dtype=np.float32)
    z0 = np.arange(bbox_min[2], bbox_max[2] + a / 2, a, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x0, y0, z0, indexing='ij')
    origins = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    offsets = np.array([
        [0, 0, 0],
        [0.5 * a, 0.5 * a, 0],
        [0.5 * a, 0, 0.5 * a],
        [0, 0.5 * a, 0.5 * a]
    ], dtype=np.float32)

    all_points = origins[:, None, :] + offsets[None, :, :]  # (N, 4, 3)
    all_points = all_points.reshape(-1, 3)

    mask = (all_points >= bbox_min - 1e-6).all(axis=1) & (all_points <=
                                                          bbox_max + 1e-6).all(axis=1)
    return all_points[mask].astype(np.float32)  # ← 关键：强制 float32


def generate_bcc_lattice(r, bbox_min, bbox_max):
    """向量化生成体心立方堆积（BCC）候选点"""
    a = 4 * r / np.sqrt(3)
    x0 = np.arange(bbox_min[0], bbox_max[0] + a / 2, a, dtype=np.float32)
    y0 = np.arange(bbox_min[1], bbox_max[1] + a / 2, a, dtype=np.float32)
    z0 = np.arange(bbox_min[2], bbox_max[2] + a / 2, a, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x0, y0, z0, indexing='ij')
    origins = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    offsets = np.array([
        [0, 0, 0],
        [0.5 * a, 0.5 * a, 0.5 * a]
    ], dtype=np.float32)

    all_points = origins[:, None, :] + offsets[None, :, :]
    all_points = all_points.reshape(-1, 3)

    mask = (all_points >= bbox_min - 1e-6).all(axis=1) & (all_points <=
                                                          bbox_max + 1e-6).all(axis=1)
    return all_points[mask].astype(np.float32)  # ← 强制 float32


def filter_points_inside_solid(scene, points, r):
    """
    保留所有粒子完全位于物体内部或表面与边界相切的点。
    条件：signed_distance <= -r （即球心到边界的距离 >= r）
    确保粒子不穿透模型边界。
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    signed_distances = scene.compute_signed_distance(
        o3d.core.Tensor(points)).numpy()
    return points[signed_distances <= -r]


def compute_packing_volumes(r):
    """返回各堆积方式单个粒子所占晶胞体积"""
    v_scc = 8 * (r ** 3)                       # = 8 r³
    v_bcc = (32 / (3 * np.sqrt(3))) * (r ** 3)  # ≈ 6.1584 r³
    v_hcp = 4 * np.sqrt(2) * (r ** 3)          # ≈ 5.65685 r³
    v_fcc = v_hcp                              # FCC 与 HCP 最密堆积密度相同
    return v_hcp, v_scc, v_fcc, v_bcc


def assign_volumes_by_type(packing_type, v_hcp, v_scc, v_fcc, v_bcc):
    """根据堆积类型分配每个粒子占据的体积"""
    volumes = np.empty_like(packing_type, dtype=np.float64)
    volumes[packing_type == 0] = v_hcp  # HCP
    volumes[packing_type == 1] = v_scc  # SCC
    volumes[packing_type == 2] = v_fcc  # FCC
    volumes[packing_type == 3] = v_bcc  # BCC
    return volumes


def save_to_vtp(centers, radii, volumes, types, output_path):
    """将结果保存为 VTP 文件（兼容 ParaView / PyVista）"""
    cloud = pv.PolyData(centers)
    cloud.point_data["radius"] = radii
    cloud.point_data["occupied_volume"] = volumes
    cloud.point_data["packing_type"] = types  # 0=HCP, 1=SCC, 2=FCC, 3=BCC
    cloud.save(output_path)


def visualize_points_and_mesh(centers, types, mesh, r):
    """使用 Open3D 可视化点云和原始网格"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    colors = np.zeros((len(centers), 3))
    colors[types == 0] = [1.0, 0.3, 0.3]   # HCP: 红
    colors[types == 1] = [0.3, 0.6, 1.0]   # SCC: 蓝
    colors[types == 2] = [0.3, 1.0, 0.3]   # FCC: 绿
    colors[types == 3] = [1.0, 0.7, 0.0]   # BCC: 橙
    pcd.colors = o3d.utility.Vector3dVector(colors)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([pcd, mesh], window_name=f"r={r}")


def main():
    # ==================== 配置参数 ====================
    # ply_file = r".\xyzrgb_statuette.ply"
    # ply_file = r".\11.ply"
    r = 10.0            # 粒子半径
    packing = "fcc"    # 可选: "hcp" / "scc" / "fcc" / "bcc"
    output_vtp = f".\粒子填充_r{r}_{packing}.vtp"
    # ================================================

    # 1. 读取并预处理网格
    # mesh = read_mesh_and_check(ply_file)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=200.0)
    # mesh = o3d.geometry.TriangleMesh.create_box(
    #     width=400.0, height=400.0, depth=100.0)
    # mesh = o3d.geometry.TriangleMesh.create_cylinder(
    #     radius=150.0, height=4*r)
    scene = build_ray_casting_scene(mesh)

    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.min_bound) - 2 * r
    bbox_max = np.asarray(bbox.max_bound) + 2 * r

    # 2. 初始化结果容器
    all_centers = []
    all_types = []  # 0=HCP, 1=SCC, 2=FCC, 3=BCC

    # 3. 生成并过滤候选点
    v_hcp, v_scc, v_fcc, v_bcc = compute_packing_volumes(r)

    if packing == "hcp":
        print("正在生成 HCP 密排...")
        candidates = generate_hcp_lattice(r, bbox_min, bbox_max)
        valid = filter_points_inside_solid(scene, candidates, r)
        all_centers.append(valid)
        all_types.extend([0] * len(valid))

    elif packing == "scc":
        print("正在生成 简单立方 (SCC)...")
        candidates = generate_scc_lattice(r, bbox_min, bbox_max)
        valid = filter_points_inside_solid(scene, candidates, r)
        all_centers.append(valid)
        all_types.extend([1] * len(valid))

    elif packing == "fcc":
        print("正在生成 面心立方 (FCC)...")
        candidates = generate_fcc_lattice(r, bbox_min, bbox_max)
        valid = filter_points_inside_solid(scene, candidates, r)
        all_centers.append(valid)
        all_types.extend([2] * len(valid))

    elif packing == "bcc":
        print("正在生成 体心立方 (BCC)...")
        candidates = generate_bcc_lattice(r, bbox_min, bbox_max)
        valid = filter_points_inside_solid(scene, candidates, r)
        all_centers.append(valid)
        all_types.extend([3] * len(valid))

    else:
        raise ValueError("packing 必须是 'hcp'、'scc'、'fcc'、'bcc'")

    if not all_centers or all(len(c) == 0 for c in all_centers):
        raise ValueError("没有任何粒子生成！请检查 r 是否太大")

    centers = np.vstack([c for c in all_centers if len(c) > 0])
    packing_type = np.array(all_types, dtype=np.int32)
    volumes = assign_volumes_by_type(packing_type, v_hcp, v_scc, v_fcc, v_bcc)
    total_volume = volumes.sum()

    # 4. 输出统计信息
    print("\n=== 结果汇总 ===")
    print(f"粒子总数       : {len(centers)}")

    type_names = {0: "HCP", 1: "SCC", 2: "FCC", 3: "BCC"}
    type_volumes = {0: v_hcp, 1: v_scc, 2: v_fcc, 3: v_bcc}

    unique_types = np.unique(packing_type)
    for typ in sorted(unique_types):
        count = np.sum(packing_type == typ)
        name = type_names[typ]
        vol = type_volumes[typ]
        print(f"{name} 粒子       : {count} (单个体积 {vol:.6f})")

    print(f"粒子总体积     : {total_volume:.6f}")

    # 5. 保存结果
    radii = np.full(len(centers), r, dtype=np.float32)
    save_to_vtp(centers, radii, volumes, packing_type, output_vtp)
    print(f"已保存 → {output_vtp}")

    # 6. 可视化（取消注释即可）
    # visualize_points_and_mesh(centers, packing_type, mesh, r)


if __name__ == "__main__":
    main()
