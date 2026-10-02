import geopandas as gpd
import pyarrow.parquet as pq

scales = ["10k", "50k", "100k"]
expected_counts = {"10k": 10000, "50k": 50000, "100k": 100000}

print(
    f"{'Scale':<8} | {'Target':<8} | {'GeoJSON Valid':<14} | {'Parquet Valid':<14} | {'Geometry Type':<15} | {'CRS'}"
)
print("-" * 75)

for s in scales:
    gj_path = f"datasets/palm_canopies_{s}.geojson"
    pq_path = f"datasets/palm_canopies_{s}.parquet"

    # 1. Cek GeoJSON
    gdf_gj = gpd.read_file(gj_path)
    count_gj = len(gdf_gj)
    valid_gj = gdf_gj.geometry.is_valid.all()
    geom_type = gdf_gj.geometry.geom_type.unique()[0]
    crs_str = str(gdf_gj.crs)

    # 2. Cek Parquet via Metadata PyArrow (Instant / Zero-overhead)
    pq_meta = pq.read_metadata(pq_path)
    count_pq = pq_meta.num_rows

    status_gj = (
        f"{count_gj:,} (OK)"
        if (count_gj == expected_counts[s] and valid_gj)
        else f"ERR ({count_gj})"
    )
    status_pq = (
        f"{count_pq:,} (OK)" if count_pq == expected_counts[s] else f"ERR ({count_pq})"
    )

    print(
        f"{s:<8} | {expected_counts[s]:<8} | {status_gj:<14} | {status_pq:<14} | {geom_type:<15} | {crs_str}"
    )
