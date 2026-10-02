import os
import time

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def generate_palm_canopies(n_trees, origin_x=101.5, origin_y=0.5, spacing_deg=0.00008):
    print(f"\n[*] Generating {n_trees:,} synthetic palm tree canopies...")
    start_time = time.time()

    cols = int(np.sqrt(n_trees * 1.15)) + 1
    rows = (n_trees // cols) + 1

    polygons = []
    ids = []
    crown_ids = []
    radii = []
    ndvi_values = []

    count = 0
    np.random.seed(42)  # Menjamin deterministik & reproducible

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    for r in range(rows):
        if count >= n_trees:
            break
        row_offset = (spacing_deg / 2.0) if (r % 2 == 1) else 0.0
        y = origin_y + (r * spacing_deg * np.sin(np.pi / 3))

        for c in range(cols):
            if count >= n_trees:
                break
            x = origin_x + (c * spacing_deg) + row_offset

            jit_x = x + np.random.normal(0, spacing_deg * 0.05)
            jit_y = y + np.random.normal(0, spacing_deg * 0.05)
            r_deg = np.random.uniform(0.000030, 0.000045)

            poly_x = jit_x + r_deg * np.cos(angles)
            poly_y = jit_y + r_deg * np.sin(angles)
            poly = Polygon(np.column_stack((poly_x, poly_y)))

            polygons.append(poly)
            ids.append(count + 1)
            crown_ids.append(f"PLM_{count + 1:08d}")
            radii.append(round(float(r_deg * 111320), 2))
            ndvi_values.append(round(float(np.random.uniform(0.65, 0.88)), 3))

            count += 1

    gdf = gpd.GeoDataFrame(
        {
            "id": ids,
            "crown_id": crown_ids,
            "radius_m": radii,
            "ndvi_mean": ndvi_values,
            "geometry": polygons,
        },
        crs="EPSG:4326",
    )

    print(f"[+] Selesai dibuat dalam {time.time() - start_time:.2f} detik.")
    return gdf


if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    scales = {"10k": 10_000, "50k": 50_000, "100k": 100_000}

    for label, count in scales.items():
        gdf = generate_palm_canopies(count)

        # 1. Simpan GeoJSON (Raw format)
        geojson_path = f"datasets/palm_canopies_{label}.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        raw_size = os.path.getsize(geojson_path) / (1024 * 1024)
        print(f" -> Saved: {geojson_path} ({raw_size:.2f} MB)")

        # 2. Simpan GeoParquet (ZSTD Compressed)
        parquet_path = f"datasets/palm_canopies_{label}.parquet"
        gdf.to_parquet(parquet_path, compression="zstd")
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
        ratio = parquet_size / raw_size
        print(
            f" -> Saved: {parquet_path} ({parquet_size:.2f} MB, Storage Space"
            f" Factor: {ratio:.3f})"
        )
