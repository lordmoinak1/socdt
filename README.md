# SoC-DT: Standard-of-Care Digital Twins for Tumor Dynamics

**Authors:** Moinak Bhattacharya, Gagandeep Singh, Prateek Prasanna  
**Status:** Under submission  
---

## Training

```bash
python3 train.py \
  --root /path/to/SocDT_UCSF \
  --patients_csv "/path/to/ucsf_combined.csv" \
  --lr 1e-5 \
  --weight_decay 1e-5 \
  --pid_col_img patient_id \
  --day_col day \
  --mask_col path_mask \
  --pid_col_pat SubjectID \
  --assimilate \
  --D_max 0.05 \
  --k_max 0.05 \
  --alpha 0.6 \
  --device cuda
```
---
## Citation
If you find this repository useful, please consider giving a star :star: and cite the following
```
@article{bhattacharya2025soc,
  title={SoC-DT: Standard-of-Care Aligned Digital Twins for Patient-Specific Tumor Dynamics},
  author={Bhattacharya, Moinak and Singh, Gagandeep and Prasanna, Prateek},
  journal={arXiv preprint arXiv:2510.03287},
  year={2025}
}
```
