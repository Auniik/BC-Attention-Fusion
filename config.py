import os
from utils.helpers import get_base_path


BASE_PATH = get_base_path() + '/breakhis'


FOLD_PATH = os.path.join(BASE_PATH, 'Folds.csv')
SLIDES_PATH = os.path.join(BASE_PATH, 'BreaKHis_v1/BreaKHis_v1/histology_slides/breast')