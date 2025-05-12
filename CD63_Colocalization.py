import streamlit as st
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Page config ---
st.set_page_config(page_title="EGFP-DAPI Colocalization Analysis", page_icon="🔬", layout="wide")
st.title("EGFP-DAPI Colocalization Intensity Analysis 🔬")

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

        # --- AND Operation (Colocalization Mask) ---
        colocalization_mask = egfp_mask & dapi_mask
        colocalization_intensity = (egfp_corrected * colocalization_mask).sum()

        # --- Count Nuclei in DAPI ---
        dapi_clean = morphology.remove_small_objects(dapi_mask, min_size=20)
        dapi_labels = measure.label(dapi_clean)
        dapi_count = len(measure.regionprops(dapi_labels))

        intensity_per_cell = colocalization_intensity / dapi_count if dapi_count > 0 else 0

        scaling_factor = 1000  # kilo
        scaled_coloc_intensity = colocalization_intensity / scaling_factor

        results.append({
            "Sample": sample,
            "Total Colocalization Intensity (k.a.u.)": round(scaled_coloc_intensity, 2),
            "DAPI+ Nuclei Count (cells)": dapi_count,
            "Colocalization Intensity per Cell (a.u./cell)": round(intensity_per_cell, 2)
        })


        # --- Visualization ---
        with st.expander(f"🔬 Results for {sample}"):
            fig, axes = plt.subplots(2, 2, figsize=(18, 10))

            # Normalize for display
            dapi_norm = dapi_corrected / dapi_corrected.max() if dapi_corrected.max() != 0 else dapi_corrected
            egfp_norm = egfp_corrected / egfp_corrected.max() if egfp_corrected.max() != 0 else egfp_corrected

            # DAPI image (Blue on black)
            dapi_rgb = np.dstack((np.zeros_like(dapi_norm), np.zeros_like(dapi_norm), dapi_norm))
            axes[0, 0].imshow(dapi_rgb)
            axes[0, 0].set_title("DAPI (Blue)")
            axes[0, 0].axis('off')

            # EGFP image (Green on black)
            egfp_rgb = np.dstack((np.zeros_like(egfp_norm), egfp_norm, np.zeros_like(egfp_norm)))
            axes[0, 1].imshow(egfp_rgb)
            axes[0, 1].set_title("EGFP (Green)")
            axes[0, 1].axis('off')

            # Composite overlay DAPI + EGFP (Blue + Green)
            composite_rgb = np.dstack((np.zeros_like(dapi_norm), egfp_norm, dapi_norm))
            axes[1, 0].imshow(composite_rgb)
            axes[1, 0].set_title("Composite Overlay (DAPI+EGFP)")
            axes[1, 0].axis('off')
            
            # Convert colocalization mask to float32 explicitly
            coloc_norm = colocalization_mask.astype(np.float32)

            # Normalize to 0-1 (if not empty)
            if coloc_norm.max() != 0:
                coloc_norm /= coloc_norm.max()

            # Stack RGB manually (Purple: Red + Blue channels)
            coloc_rgb = np.dstack((
                coloc_norm,                      # Red channel
                np.zeros_like(coloc_norm),       # Green channel (empty)
                coloc_norm                       # Blue channel
            ))

            # Display
            axes[1, 1].imshow(coloc_rgb)
            axes[1, 1].set_title("Colocalization Mask (Purple on Black)")
            axes[1, 1].axis('off')

            plt.tight_layout()
            st.pyplot(fig)


# --- Summary Table ---
if results:
    st.subheader("📊 Summary Table")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # CSV download button
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", data=csv, file_name="colocalization_intensity_summary.csv", mime="text/csv")
else:
    if uploaded_files:
        st.warning("⚠️ No valid EGFP + DAPI pairs found.")
    else:
        st.info("Please upload TIFF files to begin.")
