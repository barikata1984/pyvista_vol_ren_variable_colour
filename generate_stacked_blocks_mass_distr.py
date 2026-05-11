import csv

import numpy as np


def generate_gt_mass_distr():
    # generate gt mass distr ===========================================
    aabb_scale = 0.64
    density_1 = 500
    density_2 = 2000
    # voxel grid spec
    level = 7
    res = 2**level  # resolution
    half_res = res// 2
    # scaled voxel spec
    vx_side = aabb_scale / half_res
    vx_volume = vx_side**3

    # coordinates and mass distribution
    ndc_coords = (np.arange(0, res) - half_res + .5) / half_res
    coords = aabb_scale * ndc_coords

    xyzs = np.array(np.meshgrid(coords, coords, coords, indexing="ij"))  # ~ (3, res, res, res)
    xyzs = xyzs.reshape(3, -1)  # (3, N = res*res*res)

    mass_distr = np.zeros_like(xyzs[0], dtype="float")  # ~ (N, )
    # upper part - - - - - - - - - - - - - - - - - - - - -
    xidx = np.where((-.25 * aabb_scale < xyzs[0]) & (xyzs[0] < .25 * aabb_scale))
    yidx = np.where((-.5  * aabb_scale < xyzs[1]) & (xyzs[1] < .5  * aabb_scale))
    zidx = np.where(.0 < xyzs[2])
    xy_idx = np.intersect1d(xidx, yidx)
    idx1 = np.intersect1d(xy_idx, zidx)
    mass_distr[idx1] = density_1 * vx_volume
    # lower part - - - - - - - - - - - - - - - - - - - - -
    zidx = np.where(.0 > xyzs[2])
    idx2 = np.intersect1d(xy_idx, zidx)
    mass_distr[idx2] = density_2 * vx_volume
    print(f"{mass_distr.sum()=}")

    # write the scalar field into a .csv ===============================
    csv_filename = "gt_mass_distr.csv"
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        for x, y, z, mass in zip(*xyzs, mass_distr):
            writer.writerow([x, y, z, mass])

if __name__ == "__main__":
    generate_gt_mass_distr():
