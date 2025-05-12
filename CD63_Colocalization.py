import streamlit as st
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Page config ---
# --- Page config ---
st.set_page_config(page_title="EGFP-DAPI Colocalization Analysis", page_icon="🐱", layout="wide")

# --- Add fun GIF banner ---
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://media2.giphy.com/media/hXGYf4Xc9fK3JIVTTQ/giphy.gif?cid=6c09b952jt6sx8mjqj6uta4w5d6qfkw9qkwxqhjpfk42gac7&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=g" width="400">
    </div>
    """,
    unsafe_allow_html=True
)


# --- App title ---
st.title("EGFP-DAPI Colocalization Intensity Analysis 🐱")


# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload EGFP and DAPI TIFF image pairs",
    type=["tif"],
    accept_multiple_files=True
)

# --- Sample key extractor ---
def extract_sample_key(filename):
    parts = filename.replace(".tif", "").split("_")
    if len(parts) >= 3:
        return "_".join(parts[-3:])  # Customize this pattern if needed
    return None

# --- Group files by sample key ---
file_dict = defaultdict(dict)
for file in uploaded_files or []:
    fname = file.name
    key = extract_sample_key(fname)
    if key:
        if "EGFP" in fname.upper():
            file_dict[key]["EGFP"] = file
        elif "DAPI" in fname.upper():
            file_dict[key]["DAPI"] = file

if file_dict:
    st.info(f"🔍 Found {len(file_dict)} matched sample(s).")

results = []

# --- Processing ---
for sample, files in file_dict.items():
    if "EGFP" in files and "DAPI" in files:
        egfp_image = tiff.imread(files["EGFP"])
        dapi_image = tiff.imread(files["DAPI"])

        # --- Background Subtraction ---
        egfp_background = filters.gaussian(egfp_image, sigma=10)
        egfp_corrected = np.clip(egfp_image - egfp_background, 0, None)

        dapi_background = filters.gaussian(dapi_image, sigma=10)
        dapi_corrected = np.clip(dapi_image - dapi_background, 0, None)

        # --- Binary Masks ---
        egfp_mask = egfp_corrected > filters.threshold_otsu(egfp_corrected)
        dapi_mask = dapi_corrected > filters.threshold_otsu(dapi_corrected)

        # --- Colocalization Mask (AND) ---
        colocalization_mask = egfp_mask & dapi_mask
        colocalization_intensity = (egfp_corrected * colocalization_mask).sum()

        # --- DAPI Nuclei Counting ---
        dapi_clean = morphology.remove_small_objects(dapi_mask, min_size=20)
        dapi_labels = measure.label(dapi_clean)
        dapi_count = len(measure.regionprops(dapi_labels))

        # --- Scaling for Presentation (k.a.u.) ---
        scaling_factor = 1000  # kilo a.u.
        scaled_coloc_intensity = colocalization_intensity / scaling_factor

        # --- Intensity per Cell ---
        intensity_per_cell = scaled_coloc_intensity / dapi_count if dapi_count > 0 else 0

        # --- Append to Results ---
        results.append({
            "Sample": sample,
            "Total Colocalization Intensity (k.a.u.)": scaled_coloc_intensity,
            "DAPI+ Nuclei Count (cells)": dapi_count,
            "Colocalization Intensity per Cell (a.u./cell)": intensity_per_cell
        })

        # --- Visualization ---
        with st.expander(f"🔬 Results for {sample}"):
            fig, axes = plt.subplots(2, 2, figsize=(18, 10))

            # Normalize images for display
            dapi_norm = dapi_corrected / dapi_corrected.max() if dapi_corrected.max() != 0 else dapi_corrected
            egfp_norm = egfp_corrected / egfp_corrected.max() if egfp_corrected.max() != 0 else egfp_corrected

            # DAPI (Blue on Black)
            dapi_rgb = np.dstack((np.zeros_like(dapi_norm), np.zeros_like(dapi_norm), dapi_norm))
            axes[0, 0].imshow(dapi_rgb)
            axes[0, 0].set_title("DAPI (Blue)")
            axes[0, 0].axis('off')

            # EGFP (Green on Black)
            egfp_rgb = np.dstack((np.zeros_like(egfp_norm), egfp_norm, np.zeros_like(egfp_norm)))
            axes[0, 1].imshow(egfp_rgb)
            axes[0, 1].set_title("EGFP (Green)")
            axes[0, 1].axis('off')

            # Composite (Blue + Green)
            composite_rgb = np.dstack((np.zeros_like(dapi_norm), egfp_norm, dapi_norm))
            axes[1, 0].imshow(composite_rgb)
            axes[1, 0].set_title("Composite Overlay (DAPI+EGFP)")
            axes[1, 0].axis('off')

            # Colocalization (Purple on Black)
            coloc_norm = colocalization_mask.astype(np.float32)
            if coloc_norm.max() != 0:
                coloc_norm /= coloc_norm.max()
            coloc_rgb = np.dstack((coloc_norm, np.zeros_like(coloc_norm), coloc_norm))
            axes[1, 1].imshow(coloc_rgb)
            axes[1, 1].set_title("Colocalization Mask (Purple)")
            axes[1, 1].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

# --- Summary Table ---
if results:
    st.subheader("📊 Summary Table")

    # Format DataFrame for better visuals
    df_results = pd.DataFrame(results)
    df_results["Total Colocalization Intensity (k.a.u.)"] = df_results["Total Colocalization Intensity (k.a.u.)"].apply(lambda x: f"{x:,.2f}")
    df_results["Colocalization Intensity per Cell (a.u./cell)"] = df_results["Colocalization Intensity per Cell (a.u./cell)"].apply(lambda x: f"{x:,.2f}")

    st.dataframe(df_results, use_container_width=True)

    # CSV Download
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", data=csv, file_name="colocalization_intensity_summary.csv", mime="text/csv")

else:
    if uploaded_files:
        st.warning("⚠️ No valid EGFP + DAPI pairs found.")
    else:
        st.info("Please upload TIFF files to begin.")
