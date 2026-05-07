# Sanity-check generated synthetic image and mask dimensions.
# Expected: images 512x512x3 RGB jpgs, masks 512x512 P-mode pngs.

work_dir=${1:-data/gen_voc_llava}

python - <<EOF
import sys
import numpy as np
from glob import glob
from PIL import Image

work_dir = "$work_dir"
images = sorted(glob(f"{work_dir}/image/*.jpg"))
masks = sorted(glob(f"{work_dir}/mask/*.png"))
print(f"images: {len(images)}, masks: {len(masks)}")

bad = 0
for path in images[:50]:
    img = Image.open(path)
    if img.size != (512, 512) or img.mode != "RGB":
        print(f"  bad image: {path} size={img.size} mode={img.mode}")
        bad += 1
for path in masks[:50]:
    arr = np.array(Image.open(path).convert("P"))
    if arr.shape != (512, 512):
        print(f"  bad mask: {path} shape={arr.shape}")
        bad += 1

print("OK" if bad == 0 else f"{bad} bad files")
sys.exit(0 if bad == 0 else 1)
EOF
