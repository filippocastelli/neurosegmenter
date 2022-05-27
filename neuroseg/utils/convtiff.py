from skimage import io as skio
import tifffile
from pathlib import Path


dataset_path = Path("/home/castelli/repos/neuroseg/neuroseg/tests/test_datasets/single_images_1ch")
new_dataset_path = dataset_path.parent.joinpath("single_images_1ch_tif")
new_dataset_path.mkdir(exist_ok=True)

# convert all images to tif, while preserving the original file name and directory structure
for img_path in dataset_path.rglob("*.png"):
    img = skio.imread(str(img_path))
    # create new directory structure
    new_img_path = new_dataset_path.joinpath(img_path.relative_to(dataset_path))
    # change the extension to tif
    new_img_path = new_img_path.with_suffix(".tif")
    new_img_path.parent.mkdir(exist_ok=True, parents=True)
    # save image as tif
    tifffile.imsave(str(new_img_path), img)



