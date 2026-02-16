import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import tifffile as tiff

global_coms = {
    'data/KNN_New_Simulations/KNN_Right_Mid_tiff': (127.98361635368127, 128.32980878432326),
    'data/KNN_New_Simulations/KNN_Right_Up_tiff': (128.21378217578123, 128.1064649411597),
    'data/KNN_New_Simulations/KNN_Mid_Up_tiff': (128.32916216927336, 127.98298797281409),
    'data/KNN_New_Simulations/KNN_Left_Up_tiff': (128.2051107565511, 127.77572077732808),
    'data/KNN_New_Simulations/KNN_Left_Mid_tiff': (127.99353984684956, 127.56101923160612),
    'data/KNN_New_Simulations/KNN_Left_Down_tiff': (127.81147857774144, 127.76667104557288),
    'data/KNN_New_Simulations/KNN_Mid_Down_tiff': (127.5606598384147, 127.99382638449558),
    'data/KNN_New_Simulations/KNN_Right_Down_tiff': (127.81103193407083, 128.08999197025423),
}

def normalize_minmax(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def get_polarization_vector(pattern):
    """Returns dx, dy and magnitude of COM shift from center."""
    h, w = pattern.shape
    Y, X = np.indices((h, w))
    total = pattern.sum()
    if total == 0:
        return 0.0, 0.0, 0.0

    x_com = (X * pattern).sum() / total
    y_com = (Y * pattern).sum() / total

    dx = x_com - w / 2
    dy = y_com - h / 2
    mag = np.sqrt(dx**2 + dy**2)
    if mag == 0:
        return 0.0, 0.0, 0.0

    dx /= mag
    dy /= mag
    return dx, dy, mag


def get_com_deviation(pattern, average_com):
    h, w = pattern.shape
    total_intensity = pattern.sum()

    if total_intensity == 0:
        return 0.0, 0.0, 0.0


    Y, X = np.indices((h, w))
    local_com_x = (X * pattern).sum() / total_intensity
    local_com_y = (Y * pattern).sum() / total_intensity

    avg_com_y, avg_com_x = average_com
    dx = local_com_x - avg_com_x
    dy = local_com_y - avg_com_y


    mag = np.sqrt(dx**2 + dy**2)

    return dx, dy, mag


def compute_vector_field_from_folder(stem_grid, pos2path, folder):
    training_files = []
    stem_grid = normalize_minmax(stem_grid)
    grid_h, grid_w, h, w = stem_grid.shape
    u = np.zeros((grid_h, grid_w))
    v = np.zeros((grid_h, grid_w))
    m = np.zeros((grid_h, grid_w))

    for i in range(grid_h):
        for j in range(grid_w):
            pattern = stem_grid[i,j]
            dx, dy, mag = get_com_deviation(pattern, global_coms[folder])
            u[i, j] = dx * mag
            v[i, j] = dy * mag
            m[i, j] = mag

    norm_mag = normalize_magnitude(m)
    data_mean = np.mean(norm_mag)
    threshold = data_mean
    for i in range(grid_h):
        for j in range(grid_w):
            if norm_mag[i,j] > threshold:
                training_files.append(pos2path[(i, j)])
    return u, v, m, training_files

def normalize_magnitude(mag):
    """Normalize the polarization magnitude array to [0, 1]."""
    mag_min = np.min(mag)
    mag_max = np.max(mag)
    if mag_max - mag_min < 1e-8:
        return np.zeros_like(mag)
    return (mag - mag_min) / (mag_max - mag_min)


stem_data = np.zeros((128,128,256,256))

paths = ['data/KNN_New_Simulations']
all_training_files = {}
for path in paths:
    folders = os.listdir(path)
    for f in folders:
        print(f)
        folder = os.path.join(path, f)
        pos2path = {}

        for idx, filepath in enumerate(sorted(os.listdir(folder))):
            row = idx // 128
            col = idx % 128

            image = tiff.imread(os.path.join(folder,filepath)).astype(np.float32)
            stem_data[row, col] = image
            pos2path[(row, col)]  = os.path.join(folder,filepath)

        u, v, mag, training_files = compute_vector_field_from_folder(stem_data, pos2path, folder)
        print(len(training_files))
        all_training_files[folder] = training_files

        # Normalize the polarization magnitude
        mag_normalized = normalize_magnitude(mag)
joblib.dump(all_training_files, "training_files_mean_largest_magnitude.pth")

