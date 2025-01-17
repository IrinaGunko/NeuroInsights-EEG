import mne
import numpy as np
from pathlib import Path
from utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger("Beamforming")

class Beamformer:
    def __init__(self, subjects_dir, atlas_name="aparc.a2009s", subject="fsaverage"):
        self.subjects_dir = Path(subjects_dir)
        self.atlas_name = atlas_name
        self.subject = subject

    def create_forward_model(self, raw):
        try:
            logger.info("Creating forward model...")
            raw.set_montage("standard_1020")

            # Set up the source space
            logger.info("Setting up the source space with spacing='ico4' and computing distances.")
            src = mne.setup_source_space(
                subject=self.subject,
                spacing="ico4",
                surface="white",
                subjects_dir=self.subjects_dir,
                add_dist=True,
                n_jobs=4
            )

            # Create BEM model and solution
            logger.info("Creating BEM model and solution with ico=4.")
            bem = mne.make_bem_model(subject=self.subject, ico=4, subjects_dir=self.subjects_dir)
            bem_sol = mne.make_bem_solution(bem)

            # Create forward model
            logger.info("Creating the forward model...")
            fwd = mne.make_forward_solution(
                raw.info,
                trans=None,
                src=src,
                bem=bem_sol,
                eeg=True,
                meg=False
            )

            logger.info("Forward model created successfully.")
            return fwd
        except Exception as e:
            logger.error("Error creating forward model.", exc_info=True)
            raise

    def apply_beamformer(self, raw, fwd, cov=None, reg=0.05):
        try:
            logger.info("Setting average EEG reference...")
            raw.set_eeg_reference(projection=True)

            logger.info("Computing covariance matrix...")

            logger.info(f"Applying LCMV beamformer with reg={reg}...")
            filters = mne.beamformer.make_lcmv(
                raw.info,
                fwd,
                cov,
                reg=reg
            )
            stc = mne.beamformer.apply_lcmv_raw(raw, filters)

            logger.info("Beamformer applied successfully.")
            return stc
        except Exception as e:
            logger.error("Error applying beamformer.", exc_info=True)
            raise

    def map_to_atlas(self, stc, fwd):
        try:
            logger.info(f"Mapping results to atlas: {self.atlas_name}")
            labels = mne.read_labels_from_annot(
                self.subject,
                parc=self.atlas_name,
                subjects_dir=self.subjects_dir
            )
            stc_atlas = stc.extract_label_time_course(labels, fwd["src"], mode="mean")
            logger.info("Atlas mapping completed successfully.")
            return stc_atlas
        except Exception as e:
            logger.error("Error mapping to atlas.", exc_info=True)
            raise
