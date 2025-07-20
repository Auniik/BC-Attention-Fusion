import staintools
import numpy as np

class StainNormalizer:
    def __init__(self):
        self.normalizer = staintools.StainNormalizer(method='macenko')
        # Load a target image for fitting the normalizer
        target_image = staintools.read_image('./data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-027.png')
        self.normalizer.fit(target_image)

    def __call__(self, img):
        # The input is a PIL image. Convert it to a numpy array.
        img_np = np.array(img)
        # Transform the image
        img_normalized = self.normalizer.transform(img_np)
        return img_normalized