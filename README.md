# HYGD-IDSC2026
EfficientNet-B3 Quality-Aware GON Detection | IDSC 2026
# HYGD GON Detection Pipeline v6
## IDSC 2026 — International Data Science Challenge

### Overview
Automated Glaucomatous Optic Neuropathy (GON) detection from digital fundus images
using EfficientNet-B3 with quality-aware multi-task learning.

**Dataset**: Hillel Yaffe Glaucoma Dataset (HYGD) v1.1.0
**DOI**: https://doi.org/10.13026/m92s-0z95

### Results (3-Fold Cross Validation)
| Metric | Value |
|--------|-------|
| AUC-ROC (mean) | 0.9977 ± 0.001 |
| Sensitivity | 0.9872 |
| Specificity | 1.0000 |
| ECE Calibrated (5 bin, per fold) | 0.0547 |
| ECE Pooled (5 bin, 3 fold) | 0.0547 |
| Patient-Level Accuracy | 0.9922 |
| FP | 0 (tidak ada false positive) |

### Reproducing Results

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download dataset:
```bash
wget -r -N -c -np https://physionet.org/files/hillel-yaffe-glaucoma-dataset/1.1.0/
```

3. Run pipeline:
```bash
python hygd_pipeline_v6.py
```

### Architecture
- **Backbone**: EfficientNet-B3 (ImageNet pretrained)
- **Heads**: Classification (GON+/GON-) + Quality regression
- **Loss**: Focal Loss (α=0.75, γ=2.0) + Label Smoothing (ε=0.05)
- **Calibration**: Platt Scaling (LogisticRegression on val logits)
- **Augmentation**: CLAHE + ElasticTransform + GridDistortion + RandomGamma
- **CV**: 3-Fold patient-level stratified split (no leakage)
- **TTA**: 5x test-time augmentation

### Key Features (v6)
- Quality-aware model: rejects images with QS < 3.5/10
- Platt Scaling calibration (2-param, more robust than Temperature Scaling)
- Pooled ECE evaluation (5 bin, ~318 samples, statistically stable)
- Patient-level majority vote as primary clinical metric
- RandomGamma augmentation for dark/bright condition robustness

### Limitations
- Single institution: Hillel Yaffe Medical Center, Israel
- Single camera: TOPCON DRI OCT Triton, 45° FOV
- Israeli patients only (age 36-95)
- Generalization to other cameras/institutions not validated
- Test set limited (106 images, 28 GON-) — clinical claims should be verified

### Random Seed
All experiments use seed=42 for reproducibility.

### Citations
**Dataset**:
Abramovich, O., Pizem, H., Fhima, J., Berkowitz, E., Gofrit, B., Baskin, M.,
Meisel, M., Van Eijgen, J., Blumenthal, E., & Behar, J. (2026).
HYGD v1.1.0. PhysioNet. https://doi.org/10.13026/m92s-0z95

**Paper**:
Abramovich, O., et al. (2026). GONet: A Generalizable Deep Learning Model
for Glaucoma Detection.
IEEE Transactions on Biomedical Engineering, 73(1), 32-39.
https://doi.org/10.1109/TBME.2025.3576688

**Platform**:
Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation, 101(23), e215-e220.
