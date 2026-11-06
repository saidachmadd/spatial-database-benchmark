import os

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import ticker

# 1. Konfigurasi Gaya Plot Akademik IEEE
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "figure.dpi": 300,
    }
)


def visualize_micro_comparison():
    parquet_file = "datasets/palm_canopies_10k.parquet"
    geojson_file = "datasets/palm_canopies_10k.geojson"

    # Validasi keberadaan kedua berkas
    missing = [f for f in [parquet_file, geojson_file] if not os.path.exists(f)]
    if missing:
        print(
            f"[!] Error: Berkas berikut tidak ditemukan di direktori kerja: {missing}"
        )
        return

    print(f"[*] Membaca data empiris dari {parquet_file} dan {geojson_file}...")
    gdf_parquet = gpd.read_parquet(parquet_file)
    gdf_geojson = gpd.read_file(geojson_file)

    # 2. SPATIAL WINDOW QUERY (Ekstraksi Bounding Box Mikro)
    # Kita jadikan Parquet sebagai basis referensi koordinat tengah (keduanya pasti sama)
    minx, miny, maxx, maxy = gdf_parquet.total_bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    # Menentukan radius jendela pandang mikro (zoom-out proporsional ~130x130m)
    delta = 0.0006

    # Memotong blok spasial untuk kedua dataframe di koordinat yang sama persis
    subset_p = gdf_parquet.cx[cx - delta : cx + delta, cy - delta : cy + delta]
    subset_g = gdf_geojson.cx[cx - delta : cx + delta, cy - delta : cy + delta]

    print(
        f"[+] Bounding Box mengekstrak: {len(subset_p)} poligon Parquet, {len(subset_g)} poligon GeoJSON."
    )

    # 3. KANVAS KOMPARASI MIKRO BERDAMPINGAN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # --- PLOT 1 (KIRI): GeoParquet (Micro) ---
    subset_p.plot(
        ax=ax1, facecolor="#2ca02c", edgecolor="#001a00", alpha=0.9, linewidth=1.2
    )
    ax1.set_title(
        "Apache GeoParquet (.parquet)",
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Interval tick mikro yang proporsional
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.0004))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.0004))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))

    # --- PLOT 2 (KANAN): GeoJSON (Micro) ---
    subset_g.plot(
        ax=ax2, facecolor="#1f77b4", edgecolor="#001f3f", alpha=0.9, linewidth=1.2
    )
    ax2.set_title(
        "GeoJSON (.geojson)",
        fontweight="bold",
        pad=12,
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Interval tick mikro yang proporsional
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.0004))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.0004))
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))

    plt.suptitle(
        "Validasi Integritas Topologi Lintas Format Penyimpanan\n(Ekstraksi Spatial Window Box Identik)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_filename = "palm_canopies_micro_comparison.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(
        f"\n[+] Visualisasi komparasi mikro berhasil disimpan ke '{output_filename}'."
    )
    plt.show()


if __name__ == "__main__":
    visualize_micro_comparison()
