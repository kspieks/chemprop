### Introduction for included scripts
- `download_dataset.sh` can be used to automatically download datasets from Zenodo to `/paper_results/datasets`.
- `predict_DMPNN1.sh` can be used to conduct inference using D-MPNN-1 model. You may open the file and modify the arguments according to your need and setup.
- `predict_DMPNN2.sh` can be used to conduct inference using D-MPNN-2 model. You may open the file and modify the arguments according to your need and setup. For best results, we recommend that users limit their predictions to the 40 solvents used in the training of this model. A detailed list of these solvents can be found in Table 2 of the paper.
- `merge_data.py` is a helper script used in `predict_DMPNN2.sh` to merge gas phase Gibbs free energies with solvation corrections.