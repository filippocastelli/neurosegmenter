from pathlib import Path
import numpy as np
from skimage import io as skio


ds_path = Path("/home/phil/repos/neuroseg/neuroseg/tests/ds")

mask_paths= [fpath for fpath in ds_path.rglob("*.png") if (fpath.is_file() and "mask" in str(fpath))]

for mask_path in mask_paths:
    mask = skio.imread(mask_path)
    if len(np.unique(mask))!=2:
        print(mask_path)
        print(np.unique(mask))