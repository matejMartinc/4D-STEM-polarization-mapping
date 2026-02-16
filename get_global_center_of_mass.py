import os

import numpy as np
from scipy.ndimage import center_of_mass
import tifffile

def get_center_of_mass(image: np.ndarray, threshold_percentile: float = 95.0) -> np.ndarray:
    h, w = image.shape
    threshold_value = np.percentile(image, threshold_percentile)
    bright_pixels_mask = image > threshold_value
    # If no pixels are above the threshold (e.g., black image), default to center
    if np.sum(bright_pixels_mask) == 0:
        com_y, com_x = h // 2, w // 2
    else:
        # 2. Calculate the Center of Mass on the masked image
        com_y, com_x = center_of_mass(bright_pixels_mask)
    return com_y, com_x


if __name__ == "__main__":
    folder_paths = [
        'data/KNN_New_Simulations/KNN_Right_Mid_tiff',
        'data/KNN_New_Simulations/KNN_Right_Up_tiff',
        'data/KNN_New_Simulations/KNN_Mid_Up_tiff',
        'data/KNN_New_Simulations/KNN_Left_Up_tiff',
        'data/KNN_New_Simulations/KNN_Left_Mid_tiff',
        'data/KNN_New_Simulations/KNN_Left_Down_tiff',
        'data/KNN_New_Simulations/KNN_Mid_Down_tiff',
        'data/KNN_New_Simulations/KNN_Right_Down_tiff',
        'data/KNN_Experimental/KNNsc_40MX_0V_a',
        'data/KNN_Experimental/KNNsc_40MX_0V_b',
        'data/KNN_Experimental/KNNsc_40MX_0V_c',
        'data/KNN_Experimental/KNN_Janina_a',
        'data/KNN_Experimental/KNNsc_tripod_tiff',
        'data/LU_4DSTEM/LU_4DSTEM_20nm_NoTDS',
        'data/LU_4DSTEM/LU_4DSTEM_50nm_NoTD',
        'data/KNN_Right_Mid_20nm_24mrad16383',
        'data/RD_KNN_Simpl_50nm_24mrad_NoTDS16383',
        'data/RU_KNN_Simpl_20nm_24mrad_NoTDS16383',
        'data/LD_KNN_Simpl_20nm_24mrad_NoTDS',
        'data/KNN_Simpl_Right_Down_20nm',
        'data/KNN_Experimental/Simulated_anomaly',
    ]

    all_global_coms = {}
    for path in folder_paths:
        global_y, global_x = [], []
        image_paths = os.listdir(path)
        for p in image_paths:
            if p.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(path, p)
                image = tifffile.imread(image_path).astype(np.float32)
                if image.ndim > 2:
                    image = image[0]

                crop_size = 64
                com_y, com_x = get_center_of_mass(image)
                global_x.append(com_x)
                global_y.append(com_y)
        print(path, sum(global_y)/len(global_y), sum(global_x)/len(global_x))
        all_global_coms[path] = (float(sum(global_y)/len(global_y)), float(sum(global_x)/len(global_x)))
    print(all_global_coms)
