import streamlit as st
import os
import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn import plotting
from mpl_toolkits.mplot3d import Axes3D
from beamforming.beamformer import Beamformer  # Assuming this is implemented in your project
from utils.logger_manager import LoggerManager
from utils.visualization import plot_topographic_map, plot_static_brain_surface
import plotly.graph_objects as go

logger = LoggerManager.get_logger("BeamformerTab")


def process_file_with_visuals(temp_file_path, subjects_dir, output_dir, progress_bar):
    try:
        logger.info(f"Processing file: {Path(temp_file_path).name}")
        progress_bar.progress(10, "Initializing beamformer...")

        beamformer = Beamformer(subjects_dir)

        # Load raw EEG data
        raw = mne.io.read_raw_edf(temp_file_path, preload=True)
        progress_bar.progress(20, "Raw EEG data loaded.")

        # Create forward model
        fwd = beamformer.create_forward_model(raw)
        progress_bar.progress(40, "Forward model created.")

        # Apply beamformer
        stc = beamformer.apply_beamformer(raw, fwd)
        progress_bar.progress(60, "Beamformer applied.")

        # Map to Destrieux atlas
        stc_atlas = beamformer.map_to_atlas(stc, fwd)
        progress_bar.progress(80, "Mapped to Destrieux atlas.")

        # Save results
        stc_atlas_path = Path(output_dir) / f"{Path(temp_file_path).stem}_stc_atlas.npy"
        np.save(stc_atlas_path, stc_atlas)
        progress_bar.progress(90, "Results saved.")

        return stc, stc_atlas, stc_atlas_path
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        st.error(f"Error processing file: {e}")
        return None, None, None


def generate_visuals(stc, stc_atlas, subjects_dir):
    """Generate visualizations using Plotly, Nilearn, and Matplotlib."""
    # Replace with actual data or process outputs
    vertices = np.random.rand(100, 3)  # Example vertices
    faces = np.random.randint(0, 100, (300, 3))  # Example faces
    scalars = np.random.rand(100)  # Example scalars for Plotly
    atlas_regions = np.random.randint(0, 10, len(vertices))  # Example atlas regions for Nilearn

    # Plotly Visualization
    try:
        st.info("Generating Plotly visualization...")
        x, y, z = vertices.T
        i, j, k = faces.T
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k, intensity=scalars,
                    colorscale="Viridis", opacity=0.7
                )
            ]
        )
        fig.update_layout(title="Plotly Brain Surface Visualization")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Plotly visualization failed: {e}")

    # Nilearn Visualization
    try:
        st.info("Generating Nilearn visualization...")
        surf_mesh = {"coordinates": vertices, "faces": faces}
        plotting.plot_surf_roi(
            surf_mesh=surf_mesh, roi_map=atlas_regions,
            hemi='left', view='lateral', title="Nilearn Visualization", bg_map=None
        )
        plotting.show()
    except Exception as e:
        st.error(f"Nilearn visualization failed: {e}")

    # Matplotlib Visualization
    try:
        st.info("Generating Matplotlib visualization...")
        if len(vertices) < 3 or len(faces) < 1:
            raise ValueError("Insufficient data for Matplotlib visualization.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for face in faces:
            triangle = vertices[face]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color="blue", alpha=0.3)
        plt.title("Matplotlib Brain Surface Visualization")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Matplotlib visualization failed: {e}")


def render():
    st.title("Beamformer and Atlas Mapping")

    if "temp_file_path" not in st.session_state:
        st.error("No file uploaded or preloaded. Please upload a file in the 'Upload EEG File' tab.")
        return

    subjects_dir = "subjects_dir"  # Path to Freesurfer subjects directory
    output_dir = "beamforming_results"
    os.makedirs(output_dir, exist_ok=True)

    st.info("Process the uploaded/preloaded EEG file using beamformer and Destrieux atlas mapping.")
    if st.button("Run Beamformer and Atlas Mapping"):
        temp_file_path = st.session_state.temp_file_path

        if temp_file_path and os.path.exists(temp_file_path):
            progress_bar = st.progress(0, "Starting processing...")
            stc, stc_atlas, stc_atlas_path = process_file_with_visuals(temp_file_path, subjects_dir, output_dir, progress_bar)

            if stc is not None and stc_atlas is not None and stc_atlas.size > 0:
                progress_bar.progress(100, "Processing complete.")
                st.success("Processing completed successfully.")

                # Generate visualizations
                generate_visuals(stc, stc_atlas, subjects_dir)

                # Allow file download
                with open(stc_atlas_path, "rb") as f:
                    data = f.read()
                st.download_button(
                    label="Download Processed Atlas File",
                    data=data,
                    file_name=os.path.basename(stc_atlas_path),
                    mime="application/octet-stream",
                )
            else:
                st.error("Beamformer or atlas mapping returned empty results.")
        else:
            st.error("No valid file found to process.")
