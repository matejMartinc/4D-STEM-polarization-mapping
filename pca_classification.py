import os

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import albumentations as A
import tifffile
import cv2

global_coms = {
    'data/KNN_New_Simulations/KNN_Right_Mid_tiff': (127.98361635368127, 128.32980878432326),
    'data/KNN_New_Simulations/KNN_Right_Up_tiff': (128.21378217578123, 128.1064649411597),
    'data/KNN_New_Simulations/KNN_Mid_Up_tiff': (128.32916216927336, 127.98298797281409),
    'data/KNN_New_Simulations/KNN_Left_Up_tiff': (128.2051107565511, 127.77572077732808),
    'data/KNN_New_Simulations/KNN_Left_Mid_tiff': (127.99353984684956, 127.56101923160612),
    'data/KNN_New_Simulations/KNN_Left_Down_tiff': (127.81147857774144, 127.76667104557288),
    'data/KNN_New_Simulations/KNN_Mid_Down_tiff': (127.5606598384147, 127.99382638449558),
    'data/KNN_New_Simulations/KNN_Right_Down_tiff': (127.81103193407083, 128.08999197025423),
    'data/KNN_Experimental/KNNsc_40MX_0V_a': (120.83684409295618, 127.36708076884995),
    'data/KNN_Experimental/KNNsc_40MX_0V_b': (120.83684409295552, 127.36708076885009),
    'data/KNN_Experimental/KNNsc_40MX_0V_c': (120.61415874144447, 127.52322138691976),
    'data/KNN_Experimental/KNN_Janina_a': (126.98689420542041, 123.2817309777044),
    'data/KNN_Experimental/KNNsc_tripod_tiff': (124.89649400447742, 123.22913202951223),
    'data/KNN_Experimental/Simulated_anomaly': (127.86537540588256, 128.08908346465432),
    'data/LU_4DSTEM/LU_4DSTEM_20nm_NoTDS': (128.21647437916096, 127.84473797666828),
    'data/LU_4DSTEM/LU_4DSTEM_50nm_NoTD': (127.77318551811736, 127.77334435480056),
    'data/KNN_Right_Mid_20nm_24mrad16383': (127.98361635368224, 128.32980878432403),
    'data/RD_KNN_Simpl_50nm_24mrad_NoTDS16383': (127.93535140232265, 128.10867304914018),
    'data/RU_KNN_Simpl_20nm_24mrad_NoTDS16383': (127.89713423085495, 127.89677841284399),
    'data/LD_KNN_Simpl_20nm_24mrad_NoTDS': (127.85917481704536, 127.88918756786924),
    'data/KNN_Simpl_Right_Down_20nm': (127.81103193407107, 128.0899919702541),
}

def ensure_float32(image, **kwargs):
    return image.astype(np.float32)

def crop_around_corrected_com(
    image: np.ndarray,
    crop_size: int,
    average_com: tuple[float, float],
) -> np.ndarray:
    h, w = image.shape
    avg_com_y, avg_com_x = average_com


    local_com_y, local_com_x = h / 2, w / 2
    image_center_y = h / 2
    image_center_x = w / 2

    shift_y = avg_com_y - image_center_y
    shift_x = avg_com_x - image_center_x

    corrected_com_y = local_com_y - shift_y
    corrected_com_x = local_com_x - shift_x

    start_h = int(round(corrected_com_y - crop_size / 2))
    start_w = int(round(corrected_com_x - crop_size / 2))

    start_h = max(0, start_h)
    start_w = max(0, start_w)

    if start_h + crop_size > h:
        start_h = h - crop_size
    if start_w + crop_size > w:
        start_w = w - crop_size

    # Re-clamp in case the image is smaller than the crop size.
    start_h = max(0, start_h)
    start_w = max(0, start_w)

    return image[start_h:start_h + crop_size, start_w:start_w + crop_size]

available_classes = {
    'KNN_Right_Mid_tiff': 0,
    'KNN_Right_Up_tiff': 1,
    'KNN_Mid_Up_tiff': 2,
    'KNN_Left_Up_tiff': 3,
    'KNN_Left_Mid_tiff': 4,
    'KNN_Left_Down_tiff': 5,
    'KNN_Mid_Down_tiff': 6,
    'KNN_Right_Down_tiff': 7,
}

idx2label = {
        0: 'RM',
        1: 'RU',
        2: 'MU',
        3: 'LU',
        4: 'LM',
        5: 'LD',
        6: 'MD',
        7: 'RD',
    }


def smart_blur(image, **kwargs):
    """Blur that preserves center-of-mass shifts"""
    image = image.astype(np.float32)

    # Apply MILD blur only - just enough to smooth edges
    sigma = np.random.uniform(2, 5)  # Much milder!
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    return blurred.astype(np.float32)


# Image preprocessing function
def preprocess_image(path, train_transforms):
    transforms = A.Compose([
        #comment augmentations if you do not want to augment the train dataset
        A.Lambda(image=ensure_float32, p=1.0),
        A.Lambda(image=smart_blur, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.GaussNoise(per_channel=False, p=0.2),
    ], p=1.)
    image = tifffile.imread(path).astype(np.float32)
    if image.ndim > 2:
        image = image[0]
    crop_size = 64
    idx = path.rfind('/')

    f_path = path[:idx]
    image = crop_around_corrected_com(image, crop_size=crop_size, average_com=global_coms[f_path])
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    if train_transforms:
        image = transforms(image=image)['image']  # / 255.0

    return image.flatten()  # Convert to 1D vector

def classify_new_pattern(image_path, model, pca, scaler):
    image_vector = preprocess_image(image_path, train_transforms=False)
    image_vector = scaler.transform([image_vector])  # Normalize
    image_pca = pca.transform(image_vector)  # Apply PCA

    predicted_label = model.predict(image_pca)[0]
    return predicted_label

all_synth_accuracies = []
training_files = joblib.load("training_files_mean_largest_magnitude.pth")
# Set training files to None if you do not want to filter train set
#training_files = None

for seed in [888, 1212, 3000, 5000, 7000]:
    print('Seed', seed)
    image_paths = []  # Add all images
    polarization_labels = []  # Manually label with correct polarization direction
    synth_image_folder = 'data/KNN_New_Simulations'
    folders = os.listdir(synth_image_folder)

    for folder in folders:
        image_folder = os.listdir(os.path.join(synth_image_folder, folder))
        if training_files is not None:
            training_f = set(training_files[os.path.join(synth_image_folder, folder)])
        for ip in image_folder:
            image_path = os.path.join(synth_image_folder, folder, ip)
            if training_files is not None:
                if image_path in training_f:
                    image_paths.append(os.path.join(synth_image_folder, folder, ip))
                    polarization_labels.append(available_classes[folder])
            else:
                image_paths.append(os.path.join(synth_image_folder, folder, ip))
                polarization_labels.append(available_classes[folder])

    # Create dataset
    X = np.array([preprocess_image(path, train_transforms=True) for path in image_paths])
    y = np.array(polarization_labels)

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=20, random_state=seed)  # Reduce to 20 principal components
    X_pca = pca.fit_transform(X_scaled)
    print('Pca applied')

    # Split data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=seed)

    # Train k-NN classifier
    print('Training KNN')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Test classifier
    accuracy = knn.score(X_test, y_test)
    all_synth_accuracies.append(accuracy)
    print(f"Classification Accuracy on Synthetic data: {accuracy:.2%}")
    config = 'pca_20'
    output_folder = 'results_final'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    experimental_paths = ['data/KNN_Experimental/KNN_Janina_a',
                           'data/KNN_Experimental/KNNsc_tripod_tiff',
                           'data/KNN_Experimental/KNNsc_40MX_0V_a',
                           'data/KNN_Experimental/KNNsc_40MX_0V_b',
                           'data/KNN_Experimental/KNNsc_40MX_0V_c',
                           'data/LU_4DSTEM/LU_4DSTEM_20nm_NoTDS',
                           'data/LU_4DSTEM/LU_4DSTEM_50nm_NoTD',
                           'data/KNN_Right_Mid_20nm_24mrad16383',
                           'data/RD_KNN_Simpl_50nm_24mrad_NoTDS16383',
                           'data/RU_KNN_Simpl_20nm_24mrad_NoTDS16383',
                           'data/LD_KNN_Simpl_20nm_24mrad_NoTDS',
                           'data/KNN_Simpl_Right_Down_20nm',
                           'data/KNN_Experimental/Simulated_anomaly',
    ]
    labels = ['RM', 'RM', 'RM', 'RM', 'RM', 'LU', 'LU', 'RM', 'RD', 'RU', 'LD', 'RD', 'RD']
    output_paths = [output_folder + '/' + config + '_' + str(seed) + '_KNN_Janina_a.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_KNNsc_tripod_tiff.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_KNNsc_40MX_0V_a.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_KNNsc_40MX_0V_b.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_KNNsc_40MX_0V_c.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_LU_4DSTEM_20nm_NoTDS.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_LU_4DSTEM_50nm_NoTD.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_KNN_Right_Mid_20nm_24mrad16383.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_RD_KNN_Simpl_50nm_24mrad_NoTDS16383.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_RU_KNN_Simpl_20nm_24mrad_NoTDS16383.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_LD_KNN_Simpl_20nm_24mrad_NoTDS.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_KNN_Simpl_Right_Down_20nm.tsv',
                    output_folder + '/' + config + '_' + str(seed) + '_Simulated_anomaly.tsv',
                   ]

    for experimental_path, label, output_path in zip(experimental_paths, labels, output_paths):
        # Example
        print("Predicting", experimental_path)
        results = []
        for img in os.listdir(experimental_path):
            f_path = img.split('.')[0][-5:]
            pred = classify_new_pattern(os.path.join(experimental_path, img), knn, pca, scaler)
            results.append((f_path, idx2label[pred], label))
        df = pd.DataFrame(results, columns=['df_index', 'pred', 'true'])
        df.to_csv(output_path, index=False, sep='\t')

print("All synth accuracies", all_synth_accuracies)

