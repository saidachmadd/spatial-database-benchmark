import os
import subprocess
import sys
import threading
import time

import duckdb
import geopandas as gpd
import mysql.connector
import numpy as np
import pandas as pd
import psutil
import psycopg2
from sqlalchemy import create_engine

# Cek ketersediaan docker SDK
try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# ==============================================================================
# KONFIGURASI GLOBAL & LINGKUNGAN UJI
# ==============================================================================
CONFIG = {
    "POSTGIS": {
        "host": "localhost",
        "port": 5432,
        "dbname": "geodb_bench",
        "user": "bench_user",
        "password": "bench_password",
        "container_name": "postgis_sut",
    },
    "MYSQL": {
        "host": "localhost",
        "port": 3307,
        "database": "geodb_bench",
        "user": "bench_user",
        "password": "bench_password",
        "container_name": "mysql_sut",
    },
    "ITERATIONS": 30,
    "RANDOM_SEED": 42,
}


# ==============================================================================
# 1. DOCKER & OS RESOURCE PROFILER (ISO/IEC 25010)
# ==============================================================================
class DockerMemoryTracker:
    """Melacak puncak alokasi memori fisik (Peak RSS) kontainer daemon DB secara background."""

    def __init__(self, container_name):
        self.container_name = container_name
        self.peak_rss_mb = 0.0
        self._is_running = False
        self.client = None
        self.container = None

        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
                self.container = self.client.containers.get(container_name)
            except Exception as e:
                print(f"[WARN] Tidak dapat mengakses kontainer '{container_name}': {e}")

    def _monitor(self):
        while self._is_running:
            try:
                stats = self.container.stats(stream=False)
                # Ambil usage dari cgroup memory_stats
                mem_bytes = stats["memory_stats"].get("usage", 0)
                mem_mb = mem_bytes / (1024 * 1024)
                self.peak_rss_mb = max(self.peak_rss_mb, mem_mb)
            except Exception:
                pass
            time.sleep(0.05)

    def start(self):
        if self.container:
            self.peak_rss_mb = 0.0
            self._is_running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def stop(self):
        if self.container and self._is_running:
            self._is_running = False
            self.thread.join(timeout=1.0)
        return round(self.peak_rss_mb, 2)


# ==============================================================================
# 2. PROTOKOL DUAL-LAYER CACHE ISOLATION
# ==============================================================================
def drop_all_caches(sut_name, db_conn=None):
    """
    Layer 1: OS Kernel Drop Caches (Linux/WSL)
    Layer 2: Engine Internal Buffer Invalidation
    """
    # Layer 1: OS Pagecache Flush
    try:
        if sys.platform.startswith("linux"):
            subprocess.run(
                ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        # Fallback silent warning jika dijalankan di Windows native tanpa sudo
        pass

    # Layer 2: Database Buffer Flush
    if sut_name == "POSTGIS" and db_conn:
        try:
            with db_conn.cursor() as cur:
                cur.execute("DISCARD ALL;")
                db_conn.commit()
        except Exception:
            pass
    elif sut_name == "MYSQL" and db_conn:
        try:
            with db_conn.cursor() as cur:
                cur.execute("FLUSH TABLES;")
                db_conn.commit()
        except Exception:
            pass


# ==============================================================================
# 3. DETERMINISTIC WORKLOAD SAMPLING (SEEDED RANDOM BBOXES)
# ==============================================================================
def generate_sample_bboxes(parquet_path, n=30, window_deg=0.003, seed=42):
    """Menghasilkan rentang bounding box acak terdistribusi secara deterministik."""
    np.random.seed(seed)
    gdf_sample = gpd.read_parquet(parquet_path, columns=["geometry"])
    minx, miny, maxx, maxy = gdf_sample.total_bounds

    bboxes = []
    for _ in range(n):
        rx = np.random.uniform(minx, maxx - window_deg)
        ry = np.random.uniform(miny, maxy - window_deg)
        bboxes.append(
            {"minx": rx, "miny": ry, "maxx": rx + window_deg, "maxy": ry + window_deg}
        )
    return bboxes


# ==============================================================================
# 4. BENCHMARK SUITE: DUCKDB SPATIAL
# ==============================================================================
def benchmark_duckdb(scale_label, bboxes):
    print("\n==================================================")
    print(f"[*] [SUT 1/3] BENCHMARK DUCKDB SPATIAL: {scale_label.upper()}")
    print("==================================================")
    process = psutil.Process(os.getpid())
    parquet_file = f"datasets/palm_canopies_{scale_label}.parquet"
    db_file = f"datasets/duckdb_{scale_label}.db"

    if os.path.exists(db_file):
        os.remove(db_file)

    con = duckdb.connect(database=db_file)
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    con.execute("SET threads TO 4;")
    con.execute("SET memory_limit = '1GB';")

    # Ingestion murni Parquet
    t0 = time.perf_counter()
    con.execute(
        f"CREATE TABLE palm_canopies AS SELECT * FROM read_parquet('{parquet_file}');"
    )
    ingest_sec = time.perf_counter() - t0

    features_count = con.execute("SELECT count(*) FROM palm_canopies;").fetchone()[0]
    throughput = features_count / ingest_sec

    # Spatial Index Build
    t_idx = time.perf_counter()
    con.execute("CREATE INDEX idx_duck_geom ON palm_canopies USING RTREE (geometry);")
    index_sec = time.perf_counter() - t_idx

    peak_rss = process.memory_info().rss / (1024 * 1024)
    con.close()

    disk_size_mb = os.path.getsize(db_file) / (1024 * 1024)

    # Evaluasi Latensi Cold vs Warm
    cold_latencies = []
    warm_latencies = []

    # Cold Runs (Tutup dan Buka Koneksi File DB di setiap iterasi untuk invalidasi cache memory)
    for bbox in bboxes:
        drop_all_caches("DUCKDB")
        con = duckdb.connect(database=db_file)
        con.execute("LOAD spatial;")
        con.execute("SET threads TO 4;")
        q = f"""SELECT count(*) FROM palm_canopies WHERE ST_Intersects(geometry, 
                ST_MakeEnvelope({bbox["minx"]}, {bbox["miny"]}, {bbox["maxx"]}, {bbox["maxy"]}));"""
        t_q = time.perf_counter()
        con.execute(q).fetchall()
        cold_latencies.append((time.perf_counter() - t_q) * 1000)
        con.close()

    # Warm Runs (Single session persisten)
    con = duckdb.connect(database=db_file)
    con.execute("LOAD spatial;")
    con.execute("SET threads TO 4;")
    # 1x Warm-up run
    b0 = bboxes[0]
    con.execute(f"""SELECT count(*) FROM palm_canopies WHERE ST_Intersects(geometry, 
                    ST_MakeEnvelope({b0["minx"]}, {b0["miny"]}, {b0["maxx"]}, {b0["maxy"]}));""").fetchall()

    for bbox in bboxes:
        q = f"""SELECT count(*) FROM palm_canopies WHERE ST_Intersects(geometry, 
                ST_MakeEnvelope({bbox["minx"]}, {bbox["miny"]}, {bbox["maxx"]}, {bbox["maxy"]}));"""
        t_q = time.perf_counter()
        con.execute(q).fetchall()
        warm_latencies.append((time.perf_counter() - t_q) * 1000)
    con.close()

    return {
        "SUT": "DuckDB Spatial",
        "Scale": scale_label,
        "Features": features_count,
        "Ingest_Time_Sec": round(ingest_sec, 4),
        "Throughput_fps": round(throughput, 2),
        "Index_Build_Sec": round(index_sec, 4),
        "Disk_Size_MB": round(disk_size_mb, 2),
        "Peak_RSS_MB": round(peak_rss, 2),
        "Cold_p50_ms": round(float(np.percentile(cold_latencies, 50)), 2),
        "Cold_p95_ms": round(float(np.percentile(cold_latencies, 95)), 2),
        "Cold_p99_ms": round(float(np.percentile(cold_latencies, 99)), 2),
        "Warm_p50_ms": round(float(np.percentile(warm_latencies, 50)), 2),
        "Warm_p95_ms": round(float(np.percentile(warm_latencies, 95)), 2),
        "Warm_p99_ms": round(float(np.percentile(warm_latencies, 99)), 2),
    }


# ==============================================================================
# 5. BENCHMARK SUITE: POSTGIS 16 (BINARY PROTOCOL)
# ==============================================================================
def benchmark_postgis(scale_label, bboxes):
    print("\n==================================================")
    print(f"[*] [SUT 2/3] BENCHMARK POSTGIS 16: {scale_label.upper()}")
    print("==================================================")
    cfg = CONFIG["POSTGIS"]
    mem_tracker = DockerMemoryTracker(cfg["container_name"])
    mem_tracker.start()

    engine_uri = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    sa_engine = create_engine(engine_uri)
    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Session Resource Parity
    cur.execute("SET max_parallel_workers_per_gather = 4;")
    cur.execute("SET work_mem = '256MB';")

    table_name = f"palm_canopies_{scale_label}"
    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

    # Baca file Parquet ke GeoDataFrame (Fair Ingestion: format sumber identik)
    parquet_file = f"datasets/palm_canopies_{scale_label}.parquet"
    gdf = gpd.read_parquet(parquet_file)
    features_count = len(gdf)

    # Ingestion via GeoPandas High-Speed Binary Stream (to_postgis)
    print(f"[*] PostGIS: Ingesti {features_count:,} baris secara biner...")
    t0 = time.perf_counter()
    gdf.to_postgis(table_name, sa_engine, if_exists="replace", index=False)
    ingest_sec = time.perf_counter() - t0
    throughput = features_count / ingest_sec

    # Spatial Index Build (GiST)
    t_idx = time.perf_counter()
    cur.execute(
        f"CREATE INDEX idx_{table_name}_geom ON {table_name} USING GIST (geometry);"
    )
    index_sec = time.perf_counter() - t_idx

    # Disk footprint
    cur.execute(f"SELECT pg_total_relation_size('{table_name}');")
    disk_size_mb = cur.fetchone()[0] / (1024 * 1024)

    # Query Evaluation: Cold vs Warm
    cold_latencies = []
    warm_latencies = []

    # Cold Runs
    for bbox in bboxes:
        drop_all_caches("POSTGIS", conn)
        q = f"""SELECT count(*) FROM {table_name} WHERE ST_Intersects(geometry, 
                ST_MakeEnvelope({bbox["minx"]}, {bbox["miny"]}, {bbox["maxx"]}, {bbox["maxy"]}, 4326));"""
        t_q = time.perf_counter()
        cur.execute(q)
        cur.fetchall()
        cold_latencies.append((time.perf_counter() - t_q) * 1000)

    # Warm Runs (1x warm-up lalu loop)
    b0 = bboxes[0]
    cur.execute(f"""SELECT count(*) FROM {table_name} WHERE ST_Intersects(geometry, 
                    ST_MakeEnvelope({b0["minx"]}, {b0["miny"]}, {b0["maxx"]}, {b0["maxy"]}, 4326));""")
    cur.fetchall()

    for bbox in bboxes:
        q = f"""SELECT count(*) FROM {table_name} WHERE ST_Intersects(geometry, 
                ST_MakeEnvelope({bbox["minx"]}, {bbox["miny"]}, {bbox["maxx"]}, {bbox["maxy"]}, 4326));"""
        t_q = time.perf_counter()
        cur.execute(q)
        cur.fetchall()
        warm_latencies.append((time.perf_counter() - t_q) * 1000)

    cur.close()
    conn.close()
    peak_rss = mem_tracker.stop()

    return {
        "SUT": "PostGIS 16",
        "Scale": scale_label,
        "Features": features_count,
        "Ingest_Time_Sec": round(ingest_sec, 4),
        "Throughput_fps": round(throughput, 2),
        "Index_Build_Sec": round(index_sec, 4),
        "Disk_Size_MB": round(disk_size_mb, 2),
        "Peak_RSS_MB": peak_rss,
        "Cold_p50_ms": round(float(np.percentile(cold_latencies, 50)), 2),
        "Cold_p95_ms": round(float(np.percentile(cold_latencies, 95)), 2),
        "Cold_p99_ms": round(float(np.percentile(cold_latencies, 99)), 2),
        "Warm_p50_ms": round(float(np.percentile(warm_latencies, 50)), 2),
        "Warm_p95_ms": round(float(np.percentile(warm_latencies, 95)), 2),
        "Warm_p99_ms": round(float(np.percentile(warm_latencies, 99)), 2),
    }


# ==============================================================================
# 6. BENCHMARK SUITE: MYSQL 8.4 SPATIAL (VECTORIZED WKB PROTOCOL)
# ==============================================================================
def benchmark_mysql(scale_label, bboxes):
    print("\n==================================================")
    print(f"[*] [SUT 3/3] BENCHMARK MYSQL 8.4 SPATIAL: {scale_label.upper()}")
    print("==================================================")
    cfg = CONFIG["MYSQL"]
    mem_tracker = DockerMemoryTracker(cfg["container_name"])
    mem_tracker.start()

    conn = mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
    )
    cur = conn.cursor()

    table_name = f"palm_canopies_{scale_label}"
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    cur.execute(f"""
        CREATE TABLE {table_name} (
            id INT NOT NULL PRIMARY KEY,
            crown_id VARCHAR(32),
            radius_m DOUBLE,
            ndvi_mean DOUBLE,
            geometry GEOMETRY NOT NULL SRID 4326
        ) ENGINE=InnoDB;
    """)
    conn.commit()

    # Baca file Parquet (Paritas sumber input)
    parquet_file = f"datasets/palm_canopies_{scale_label}.parquet"
    gdf = gpd.read_parquet(parquet_file)
    features_count = len(gdf)

    # Ingesti WKB biner (Direct C-layer serialization via Shapely .wkb)
    print(f"[*] MySQL: Ingesti {features_count:,} baris secara biner (WKB)...")
    records = []
    for row in gdf.itertuples(index=False):
        records.append(
            (row.id, row.crown_id, row.radius_m, row.ndvi_mean, row.geometry.wkb)
        )

    insert_sql = f"""INSERT INTO {table_name} (id, crown_id, radius_m, ndvi_mean, geometry) 
                     VALUES (%s, %s, %s, %s, ST_GeomFromWKB(%s, 4326, 'axis-order=long-lat'));"""

    t0 = time.perf_counter()
    batch_size = 5000
    for i in range(0, len(records), batch_size):
        cur.executemany(insert_sql, records[i : i + batch_size])
        conn.commit()
    ingest_sec = time.perf_counter() - t0
    throughput = features_count / ingest_sec

    # Spatial Index Build (R-Tree)
    t_idx = time.perf_counter()
    cur.execute(
        f"CREATE SPATIAL INDEX idx_{table_name}_geom ON {table_name}(geometry);"
    )
    conn.commit()
    index_sec = time.perf_counter() - t_idx

    # Disk Footprint
    cur.execute(f"ANALYZE TABLE {table_name};")
    cur.fetchall()  # flush buffer output analyze
    cur.execute(f"""SELECT (data_length + index_length) FROM information_schema.tables 
                    WHERE table_schema = '{cfg["database"]}' AND table_name = '{table_name}';""")
    disk_size_mb = cur.fetchone()[0] / (1024 * 1024)

    # Query Evaluation: Cold vs Warm
    cold_latencies = []
    warm_latencies = []

    # Cold Runs
    for bbox in bboxes:
        drop_all_caches("MYSQL", conn)
        bbox_poly_wkt = f"POLYGON(({bbox['minx']} {bbox['miny']}, {bbox['maxx']} {bbox['miny']}, {bbox['maxx']} {bbox['maxy']}, {bbox['minx']} {bbox['maxy']}, {bbox['minx']} {bbox['miny']}))"
        # Cold query
        q = f"""SELECT count(*) FROM {table_name} 
                WHERE ST_Intersects(geometry, ST_GeomFromText('{bbox_poly_wkt}', 4326, 'axis-order=long-lat'));"""
        t_q = time.perf_counter()
        cur.execute(q)
        cur.fetchall()
        cold_latencies.append((time.perf_counter() - t_q) * 1000)

    # Warm Runs (1x warm-up lalu loop)
    b0 = bboxes[0]
    b0_wkt = f"POLYGON(({b0['minx']} {b0['miny']}, {b0['maxx']} {b0['miny']}, {b0['maxx']} {b0['maxy']}, {b0['minx']} {b0['maxy']}, {b0['minx']} {b0['miny']}))"
    cur.execute(
        f"SELECT count(*) FROM {table_name} WHERE ST_Intersects(geometry, ST_GeomFromText('{b0_wkt}', 4326, 'axis-order=long-lat'));"
        ""
    )
    cur.fetchall()

    for bbox in bboxes:
        bbox_poly_wkt = f"POLYGON(({bbox['minx']} {bbox['miny']}, {bbox['maxx']} {bbox['miny']}, {bbox['maxx']} {bbox['maxy']}, {bbox['minx']} {bbox['maxy']}, {bbox['minx']} {bbox['miny']}))"
        q = (
            f"SELECT count(*) FROM {table_name} WHERE ST_Intersects(geometry, ST_GeomFromText('{bbox_poly_wkt}', 4326, 'axis-order=long-lat'));"
            ""
        )
        t_q = time.perf_counter()
        cur.execute(q)
        cur.fetchall()
        warm_latencies.append((time.perf_counter() - t_q) * 1000)

    cur.close()
    conn.close()
    peak_rss = mem_tracker.stop()

    return {
        "SUT": "MySQL 8.4 Spatial",
        "Scale": scale_label,
        "Features": features_count,
        "Ingest_Time_Sec": round(ingest_sec, 4),
        "Throughput_fps": round(throughput, 2),
        "Index_Build_Sec": round(index_sec, 4),
        "Disk_Size_MB": round(disk_size_mb, 2),
        "Peak_RSS_MB": peak_rss,
        "Cold_p50_ms": round(float(np.percentile(cold_latencies, 50)), 2),
        "Cold_p95_ms": round(float(np.percentile(cold_latencies, 95)), 2),
        "Cold_p99_ms": round(float(np.percentile(cold_latencies, 99)), 2),
        "Warm_p50_ms": round(float(np.percentile(warm_latencies, 50)), 2),
        "Warm_p95_ms": round(float(np.percentile(warm_latencies, 95)), 2),
        "Warm_p99_ms": round(float(np.percentile(warm_latencies, 99)), 2),
    }


# ==============================================================================
# 7. CONTROLLER MASTER
# ==============================================================================
if __name__ == "__main__":
    scales = ["10k", "50k", "100k"]
    master_results = []

    print("\n================================================================")
    print("[*] MEMULAI REFACTORED BENCHMARK HARNESS (IEEE COMPLIANT)")
    print("================================================================")

    for s in scales:
        parquet_path = f"datasets/palm_canopies_{s}.parquet"
        if not os.path.exists(parquet_path):
            print(
                f"[!] File {parquet_path} tidak ditemukan. Buat dataset terlebih dahulu."
            )
            continue

        # Bentuk 30 BBOX dinamis berbasis seed 42
        bboxes = generate_sample_bboxes(
            parquet_path, n=CONFIG["ITERATIONS"], seed=CONFIG["RANDOM_SEED"]
        )

        # 1. Eksekusi DuckDB
        res_duck = benchmark_duckdb(s, bboxes)
        master_results.append(res_duck)

        # 2. Eksekusi PostGIS 16
        try:
            res_postgis = benchmark_postgis(s, bboxes)
            master_results.append(res_postgis)
        except Exception as e:
            print(f"[!] PostGIS [{s}] gagal / kontainer belum aktif: {e}")

        # 3. Eksekusi MySQL 8.4 Spatial
        try:
            res_mysql = benchmark_mysql(s, bboxes)
            master_results.append(res_mysql)
        except Exception as e:
            print(f"[!] MySQL [{s}] gagal / kontainer belum aktif: {e}")

    # Ekspor Rekapitulasi Metrik
    df_master = pd.DataFrame(master_results)
    os.makedirs("results", exist_ok=True)
    out_csv = "results/master_metrics.csv"
    df_master.to_csv(out_csv, index=False)

    print("\n================================================================")
    print("[*] HASIL AKHIR REKAPITULASI DUAL-LAYER METRICS:")
    print("================================================================")
    print(df_master.to_string(index=False))
    print(f"\n[+] Sukses disimpan ke: {out_csv}")
