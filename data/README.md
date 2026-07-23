# Data


## Download fastMRI Dataset


The project uses the **fastMRI single-coil knee** dataset.


1. Register and download from: https://fastmri.med.nyu.edu/
2. Select: `Knee MRI` → `Single-coil`
3. Place the `.h5` files into:


```
data/
├── knee_singlecoil_train/    ← training volumes (.h5)
└── knee_singlecoil_val/      ← validation volumes (.h5)
```


## Dataset Statistics


| Split | Volumes |
|---|---|
| Training | 973 |
| Validation | 199 |


## Preprocessing Summary


Each `.h5` file contains multi-slice complex k-space data.
The pipeline:
1. Extracts the **middle slice** from each volume
2. Applies a **random undersampling mask** (center_fraction=0.08, acceleration=4×)
3. Reconstructs via **inverse FFT** (zero-filled)
4. Normalizes by per-scan maximum
5. Center-crops to **320×320 pixels**