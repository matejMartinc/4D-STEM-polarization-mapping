import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# Configuration
METHODS = ['conv_just_contrastive', 'pca_20']

results_folder = "results_final_filtering_preprocessing"


#Visualize results for experimental images
experiments = ['KNNsc_40MX_0V_a',
               'KNNsc_40MX_0V_b',
               'KNNsc_40MX_0V_c',
               'KNN_Janina_a',
               'KNNsc_tripod_tiff']

#visualize results for synthetic images
# experiments = ['LU_4DSTEM_20nm_NoTDS',
#                'KNN_Right_Mid_20nm_24mrad16383',
#                'RU_KNN_Simpl_20nm_24mrad_NoTDS16383',
#                'LD_KNN_Simpl_20nm_24mrad_NoTDS',
#                'KNN_Simpl_Right_Down_20nm',
#                'LU_4DSTEM_50nm_NoTD',
#                'RD_KNN_Simpl_50nm_24mrad_NoTDS16383']



NUM_SEEDS = 5
H, W = 128, 128  # Map size

# Class Definitions
CLASS_LABELS = ['LU', 'RU', 'MD', 'LD', 'RD', 'MU', 'RM', 'LM']
class_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
cmap = ListedColormap(class_colors)
label_to_index = {label: idx for idx, label in enumerate(CLASS_LABELS)}

# Display Name Mappings
model_display_names = {
    'conv_regression': 'Conv (Reg)',
    'conv_classification': 'Conv (Cls)',
    'conv_just_contrastive': 'Conv (Proto)',
    'vgg_regression': 'VGG (Reg)',
    'vgg_classification': 'VGG (Cls)',
    'vgg_just_contrastive': 'VGG (Proto)',
    'res_net_regression': 'ResNet (Reg)',
    'res_net_classification': 'ResNet (Cls)',
    'res_net_just_contrastive': 'ResNet (Proto)',
    'pca_20': 'PCA'
}


dataset_display_names = {
    'KNNsc_40MX_0V_a': '1E-RM',
    'KNNsc_40MX_0V_b': '2E-RM',
    'KNNsc_40MX_0V_c': '3E-RM',
    'KNN_Janina_a': '4E-RM/RD',
    'KNNsc_tripod_tiff': '5E-RM/RD',
    'Simulated_anomaly': 'Sim Anomaly',
    'LU_4DSTEM_20nm_NoTDS': '1S-LU-6x6-20nm',
    'LU_4DSTEM_50nm_NoTD': '2S-LU-6x6-50nm',
    'KNN_Right_Mid_20nm_24mrad16383': '3S-RM-2x2-20nm',
    'RD_KNN_Simpl_50nm_24mrad_NoTDS16383': '4S-RD-6x6-50nm',
    'RU_KNN_Simpl_20nm_24mrad_NoTDS16383': '5S-RU-6x6-20nm',
    'LD_KNN_Simpl_20nm_24mrad_NoTDS': '6S-LD-4x4-20nm',
    'KNN_Simpl_Right_Down_20nm': '7S-RD-2x2-20nm',
}


all_maps = {}
seeds = [888, 1212, 3000, 5000, 7000]

# --- 2. Process Data (Majority Voting and Map Generation) ---
print("Processing data and generating prediction maps...")

for method in METHODS:
    for experiment in experiments:
        idx_preds = defaultdict(list)

        # 1. Collect predictions across all seeds
        for seed in seeds:
            filename = f"{results_folder}/{method}_{seed}_{experiment}.tsv"
            try:
                df = pd.read_csv(filename, sep='\t')
            except FileNotFoundError:
                print(f"File not found: {filename}. Skipping...")
                continue

            if 'df_index' not in df.columns or 'pred' not in df.columns:
                continue

            for idx, row in df.iterrows():
                idx_preds[row['df_index']].append(row['pred'])

        # 2. Determine the final prediction (Majority Vote) for each pixel
        if not idx_preds:
            print(f"No predictions found for {method} - {experiment}. Filling map with -1.")
            class_map = np.full((H, W), -1)
            all_maps[(method, experiment)] = class_map
            continue

        final_preds = []
        for df_idx, val in idx_preds.items():
            # Find the most frequent predicted class
            pred = max(set(val), key=val.count)
            final_preds.append((df_idx, pred))

        final_preds = sorted(final_preds, key=lambda x: x[0])

        # 3. Reconstruct the 128x128 class map
        class_map = np.full((H, W), -1)
        possible_labels = list(label_to_index.keys())

        for idx, pred_label in final_preds:
            row = idx // W
            col = idx % W
            if 0 <= row < H and 0 <= col < W:
                if pred_label in label_to_index:
                    class_map[row, col] = label_to_index[pred_label]
                elif isinstance(pred_label, (int, float)):
                    int_pred = int(pred_label)
                    # Convert number to string label for mapping
                    if 0 <= int_pred < len(possible_labels):
                        label_string = possible_labels[int_pred]
                        if label_string in label_to_index:
                            class_map[row, col] = label_to_index[label_string]

        all_maps[(method, experiment)] = class_map

print("Data processing complete. Generating visualization grid...")

# --- 3. Visualization Grid Generation (Modified) ---

N_METHODS = len(METHODS)
N_EXPERIMENTS = len(experiments)

# Set up the figure and axes
# Use a slightly smaller figure size and increase the right margin for the legend
fig, axes = plt.subplots(N_METHODS, N_EXPERIMENTS,
                         figsize=(3.0 * N_EXPERIMENTS, 3.0 * N_METHODS),  # Reduced size
                         squeeze=False)

# Plotting loop
for i, method in enumerate(METHODS):
    for j, experiment in enumerate(experiments):
        ax = axes[i, j]
        class_map = all_maps.get((method, experiment), np.full((H, W), -1))  # Safely retrieve map or empty array

        # Plot the map
        ax.imshow(class_map, cmap=cmap, interpolation="nearest",
                  vmin=0, vmax=len(CLASS_LABELS) - 1)

        ax.axis("off")  # Turn off axes for clean visualization

        if i == 0:
            display_name = dataset_display_names.get(experiment, experiment)
            ax.set_title(display_name, fontsize=14, pad=5)

        # Row Labels (Method names) - set only for the first column
        if j == 0:
            display_name = model_display_names.get(method, method)
            # Add method name on the left of the subplot row
            ax.text(-0.15, 0.5, display_name,
                    transform=ax.transAxes,
                    fontsize=14,
                    va='center',
                    ha='right',
                    rotation='vertical')


fig.tight_layout()
fig.subplots_adjust(right=0.90, wspace=0.05, hspace=0.05)

# 2. Create the legend elements
legend_elements = [Patch(facecolor=class_colors[k], edgecolor='black', label=CLASS_LABELS[k])
                   for k in range(len(CLASS_LABELS))]

fig.legend(handles=legend_elements,
           loc='center right',
           bbox_to_anchor=(0.96, 0.5),  # Anchor position adjusted by fig.subplots_adjust
           fontsize=14,
           handlelength=1.0,  # Make color patches shorter
           handletextpad=0.5,  # Reduce space between patch and label
           labelspacing=0.2)  # Reduce vertical spacing between items

# Final save
output_filename = "visualization_grid_predictions.png"
plt.savefig(output_filename, bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"Grid visualization saved to {output_filename}")

