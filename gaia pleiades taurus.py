import warnings
from pathlib import Path
 
import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
 
warnings.filterwarnings("ignore")

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)
 
CLUSTER_COLORS     = ["#3A7EBF", "#E07B39", "#3DAA6E", "#9B59B6", "#E74C3C"]
PLEIADES_PMRA_REF  =  19.99  # mas/yr, Gaia Collaboration 2023
PLEIADES_PMDEC_REF = -45.52  # mas/yr
 
plt.rcParams.update({
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.labelsize"   : 11,
    "axes.titlesize"   : 12,
})

# 1. Query Data
adql = """
SELECT
    source_id, ra, dec,
    parallax, parallax_over_error,
    pmra, pmdec,
    phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE ra  BETWEEN 50 AND 75
  AND dec BETWEEN 15 AND 35
  AND parallax BETWEEN 4 AND 12
  AND parallax_over_error > 5
  AND pmra  IS NOT NULL
  AND pmdec IS NOT NULL
"""

job = Gaia.launch_job_async(adql)
df  = job.get_results().to_pandas()

# 2. EDA

# distribusi 6 fitur utama
cols   = ["ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag"]
labels = ["RA (deg)", "Dec (deg)", "Parallax (mas)",
          "PM RA (mas/yr)", "PM Dec (mas/yr)", "G Magnitude"]
 
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Gaia DR3 - Feature Distributions", fontsize=13)
 
for ax, col, label in zip(axes.flatten(), cols, labels):
    ax.hist(df[col].dropna(), bins=60, color="#3A7EBF", edgecolor="none", alpha=0.85)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(label)
 
plt.tight_layout()
plt.savefig(MEDIA_DIR / "eda_distributions.png", dpi=150, bbox_inches="tight")
 
# sky plot, warna = parallax
fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(
    df["ra"], df["dec"], s=0.4, alpha=0.4,
    c=df["parallax"], cmap="plasma",
    vmin=df["parallax"].quantile(0.02),
    vmax=df["parallax"].quantile(0.98),
)
fig.colorbar(sc, ax=ax, label="Parallax (mas)", pad=0.01)
ax.annotate("Pleiades", xy=(56.6, 24.1), fontsize=9, color="white", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="navy", alpha=0.85))
ax.annotate("Taurus", xy=(65.0, 28.0), fontsize=9, color="white", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="darkred", alpha=0.85))
ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title("Sky Distribution - Pleiades + Taurus Region")
plt.tight_layout()
plt.savefig(MEDIA_DIR / "eda_sky.png", dpi=150, bbox_inches="tight")


# 3. PREPROCESSING

df_clean = df.dropna().copy()

# filter outlier parallax 3-sigma
mu, sigma = df_clean["parallax"].mean(), df_clean["parallax"].std()
df_clean  = df_clean[np.abs(df_clean["parallax"] - mu) < 3 * sigma].reset_index(drop=True)


# 4. KOORDINAT TRANSFORM -> Cartesian (parsec)

dist     = 1000.0 / df_clean['parallax']   # mas → parsec (1/parallax dalam arcsec = 1000/parallax_mas)
ra_rad   = np.radians(df_clean['ra'])
dec_rad  = np.radians(df_clean['dec'])

df_clean['x'] = dist * np.cos(dec_rad) * np.cos(ra_rad)
df_clean['y'] = dist * np.cos(dec_rad) * np.sin(ra_rad)
df_clean['z'] = dist * np.sin(dec_rad)

# 5. FEATURE CONSTRUCTION & SCALING

features = ["x", "y", "z", "pmra", "pmdec"]
X        = df_clean[features].values
X_scaled = RobustScaler().fit_transform(X)

# 6. HDBSCAN Clustering

clusterer = hdbscan.HDBSCAN(
    min_cluster_size         = 80,
    min_samples              = 15,
    cluster_selection_epsilon= 0.3,
    metric                   = "euclidean",
    cluster_selection_method = "eom",  # EOM lebih stabil dari 'leaf' untuk densitas bervariasi
)
labels = clusterer.fit_predict(X_scaled)
df_clean["cluster"] = labels
 
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int((labels == -1).sum())
noise_pct  = n_noise / len(labels) * 100
 
# 7. EVALUASI

mask = labels != -1

 
# cluster summary table
cluster_df = df_clean[df_clean["cluster"] != -1]
summary = cluster_df.groupby("cluster").agg(
    n_stars      = ("cluster",         "count"),
    mean_ra      = ("ra",              "mean"),
    mean_dec     = ("dec",             "mean"),
    mean_plx_mas = ("parallax",        "mean"),
    mean_dist_pc = ("parallax",        lambda p: round(1000 / p.mean(), 1)),
    mean_pmra    = ("pmra",            "mean"),
    mean_pmdec   = ("pmdec",           "mean"),
    mean_gmag    = ("phot_g_mean_mag", "mean"),
).sort_values("n_stars", ascending=False)
 
# 8. VISUALISASI

noise_mask = df_clean["cluster"] == -1
 
# --- sky map
fig, ax = plt.subplots(figsize=(14, 8))
ax.scatter(df_clean.loc[noise_mask, "ra"], df_clean.loc[noise_mask, "dec"],
           s=0.3, alpha=0.12, c="gray", label="Noise (field stars)")
 
for i, cid in enumerate(sorted(df_clean.loc[~noise_mask, "cluster"].unique())):
    m = df_clean["cluster"] == cid
    ax.scatter(df_clean.loc[m, "ra"], df_clean.loc[m, "dec"],
               s=4, alpha=0.85, c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
               label=f"Cluster {cid} (n={m.sum():,})")
 
ax.annotate("Pleiades\n(~141 pc)", xy=(56.6, 24.1), fontsize=9,
            color="white", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="navy", alpha=0.88))
ax.annotate("Taurus OB\n(~186 pc)", xy=(65.0, 26.5), fontsize=9,
            color="white", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="darkred", alpha=0.88))
 
ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title(f"Sky Map - {n_clusters} Clusters Found (HDBSCAN) | Pleiades + Taurus Region")
ax.legend(markerscale=4, loc="lower left")
plt.tight_layout()
plt.savefig(MEDIA_DIR / "clusters_sky.png", dpi=150, bbox_inches="tight")
 
# 3D interactive
# sample sekali
n_sample  = min(50_000, len(df_clean))
df_sample = df_clean.sample(n_sample, random_state=42)
 
fig_3d = px.scatter_3d(
    df_sample, x="x", y="y", z="z",
    color=df_sample["cluster"].astype(str),
    opacity=0.5,
    title="3D Star Distribution - Pleiades + Taurus (Parsec Space)",
    labels={"x": "X (pc)", "y": "Y (pc)", "z": "Z (pc)", "color": "Cluster"},
    color_discrete_sequence=px.colors.qualitative.Alphabet,
)
fig_3d.update_traces(marker=dict(size=1.5))
fig_3d.write_html(str(MEDIA_DIR / "clusters_3d.html"))
 
# proper motion diagram
fig, ax = plt.subplots(figsize=(10, 9))
ax.scatter(df_clean.loc[noise_mask, "pmra"], df_clean.loc[noise_mask, "pmdec"],
           s=0.3, alpha=0.08, c="gray", label="Noise")
 
for i, cid in enumerate(summary.head(5).index):
    m    = df_clean["cluster"] == cid
    dist = summary.loc[cid, "mean_dist_pc"]
    ax.scatter(df_clean.loc[m, "pmra"], df_clean.loc[m, "pmdec"],
               s=6, alpha=0.75, c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
               label=f"Cluster {cid} (n={m.sum():,}, ~{dist} pc)")
 
ax.axvline(PLEIADES_PMRA_REF,  color="navy", lw=1.2, ls="--", alpha=0.6,
           label=f"Pleiades PM ref (pmra={PLEIADES_PMRA_REF})")
ax.axhline(PLEIADES_PMDEC_REF, color="navy", lw=1.2, ls="--", alpha=0.6,
           label=f"(pmdec={PLEIADES_PMDEC_REF})")
 
ax.set_xlabel("PM RA (mas/yr)")
ax.set_ylabel("PM Dec (mas/yr)")
ax.set_title("Proper Motion Diagram - Top 5 Clusters")
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.legend(markerscale=3, loc="upper left")
plt.tight_layout()
plt.savefig(MEDIA_DIR / "clusters_pm.png", dpi=150, bbox_inches="tight")
 
# CMD: parallax vs G magnitude
fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(df_clean.loc[noise_mask, "parallax"], df_clean.loc[noise_mask, "phot_g_mean_mag"],
           s=0.4, alpha=0.08, c="gray", label="Noise (field stars)")
 
for i, cid in enumerate(summary.index):
    m    = df_clean["cluster"] == cid
    dist = summary.loc[cid, "mean_dist_pc"]
    ax.scatter(df_clean.loc[m, "parallax"], df_clean.loc[m, "phot_g_mean_mag"],
               s=6, alpha=0.8, c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
               label=f"Cluster {cid} (~{dist} pc)")
 
ax.axvline(1000 / 141.1, color=CLUSTER_COLORS[0], lw=1, ls=":", alpha=0.7,
           label="Pleiades ref (7.09 mas)")
ax.axvline(1000 / 185.9, color=CLUSTER_COLORS[1], lw=1, ls=":", alpha=0.7,
           label="Taurus OB ref (5.38 mas)")
 
ax.invert_yaxis()
ax.set_xlabel("Parallax (mas)")
ax.set_ylabel("G Magnitude")
ax.set_title("Color-Magnitude Diagram - Parallax vs G Magnitude")
ax.legend(markerscale=3, loc="upper right")
plt.tight_layout()
plt.savefig(MEDIA_DIR / "cmd.png", dpi=150, bbox_inches="tight")
