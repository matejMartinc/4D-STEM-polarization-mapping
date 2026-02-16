import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from collections import defaultdict, Counter


CLASS_LABELS = ['MU', 'LU', 'LM', 'LD', 'MD', 'RD', 'RM', 'RU']
CORRECT_CLASSES = {
    'KNN_Janina_a': ['RM', 'RD'],
    'KNNsc_tripod_tiff': ['RM', 'RD'],
    'KNNsc_40MX_0V_a': ['RM'],
    'KNNsc_40MX_0V_b': ['RM'],
    'KNNsc_40MX_0V_c': ['RM'],
    'LU_4DSTEM_20nm_NoTDS': ['LU'],
    'LU_4DSTEM_50nm_NoTD': ['LU'],
    'KNN_Right_Mid_20nm_24mrad16383': ['RM'],
    'RD_KNN_Simpl_50nm_24mrad_NoTDS16383': ['RD'],
    'RU_KNN_Simpl_20nm_24mrad_NoTDS16383': ['RU'],
    'LD_KNN_Simpl_20nm_24mrad_NoTDS': ['LD'],
    'KNN_Simpl_Right_Down_20nm': ['RD']
}


METHODS = ['conv_just_contrastive', 'pca_20']

#Visualize results for experimental images
experiments = ['KNNsc_40MX_0V_a',
               'KNNsc_40MX_0V_b',
               'KNNsc_40MX_0V_c',
               'KNN_Janina_a',
               'KNNsc_tripod_tiff',
               ]

#Visualize results for synthetic images
# experiments = [ 'LU_4DSTEM_20nm_NoTDS',
#               'KNN_Right_Mid_20nm_24mrad16383',
#               'RU_KNN_Simpl_20nm_24mrad_NoTDS16383',
#               'LD_KNN_Simpl_20nm_24mrad_NoTDS',
#               'KNN_Simpl_Right_Down_20nm',
#               'LU_4DSTEM_50nm_NoTD',
#               'RD_KNN_Simpl_50nm_24mrad_NoTDS16383',
#               ]

SEEDS = [888, 1212, 3000, 5000, 7000]
synth_images = 'results_final_filtering_preprocessing'


# --- Display Name Mappings ---

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

# Define colors and hatch patterns
dataset_colors = plt.cm.get_cmap('Dark2', len(experiments)).colors
model_hatches = {
    'conv_just_contrastive': '////',  # Pattern for Conv (Proto)
    'pca_20': '---'  # Pattern for PCA
}


combined_percentages = defaultdict(lambda: defaultdict(list))

for method_name in METHODS:
    for experiment in experiments:
        for seed in SEEDS:
            filename = f"{synth_images}/{method_name}_{seed}_{experiment}.tsv"
            try:
                df = pd.read_csv(filename, sep='\t')
            except FileNotFoundError:
                continue

            total_predictions = len(df)
            if total_predictions == 0:
                continue

            predicted_counts = Counter(df['pred'])
            key = (method_name, experiment)

            for cls_label in CLASS_LABELS:
                percentage = (predicted_counts.get(cls_label, 0) / total_predictions) * 100
                combined_percentages[key][cls_label].append(percentage)

combined_means = {}
combined_stds = {}

for (method, experiment), class_data in combined_percentages.items():
    mean_list = []
    std_list = []
    for cls_label in CLASS_LABELS:
        percentages_list = class_data.get(cls_label, [])
        if not percentages_list:
            mean_list.append(0.0)
            std_list.append(0.0)
        else:
            mean_list.append(np.mean(percentages_list))
            std_list.append(np.std(percentages_list))

    combined_means[(method, experiment)] = mean_list
    combined_stds[(method, experiment)] = std_list


plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(20, 8))

num_classes = len(CLASS_LABELS)
num_bars_per_group = len(METHODS) * len(experiments)  # 10 bars total

x = np.arange(num_classes)
group_width_ratio = 0.95
bar_width = group_width_ratio / num_bars_per_group

plot_keys = []
for exp in experiments:
    for method in METHODS:
        plot_keys.append((method, exp))


all_handles = []
all_labels = []


for i, (method_name, experiment_name) in enumerate(plot_keys):

    offset = (i - (num_bars_per_group - 1) / 2) * bar_width

    means = combined_means.get((method_name, experiment_name), [0.0] * num_classes)
    stds = combined_stds.get((method_name, experiment_name), [0.0] * num_classes)

    exp_index = experiments.index(experiment_name)
    color = dataset_colors[exp_index]
    hatch = model_hatches[method_name]

    allowed_class_labels = CORRECT_CLASSES.get(experiment_name, [])
    # Map those labels to their indices on the X-axis
    correct_indices = [CLASS_LABELS.index(lbl) for lbl in allowed_class_labels if lbl in CLASS_LABELS]
    rects = ax.bar(x + offset, means, bar_width,
                   yerr=stds, capsize=3,
                   color=color,
                   hatch=hatch,
                   edgecolor='white',  # Default edge color
                   linewidth=1)

    for idx in correct_indices:
        if idx < len(rects):
            rects[idx].set_edgecolor('black')
            rects[idx].set_linewidth(2.5)  # Slightly thicker to stand out against hatches



    # Dataset Handle
    dataset_label = dataset_display_names.get(experiment_name, experiment_name)
    if dataset_label not in [h.get_label() for h in all_handles if h.get_label() in dataset_display_names.values()]:
        dataset_handle = patches.Patch(facecolor=color, edgecolor='white', label=dataset_label)
        all_handles.append(dataset_handle)
        all_labels.append(dataset_label)

    # Model Handle
    model_label = model_display_names.get(method_name, method_name)
    if model_label not in [h.get_label() for h in all_handles if h.get_label() in model_display_names.values()]:
        model_handle = patches.Patch(facecolor='lightgray', edgecolor='black', hatch=hatch, label=model_label)
        all_handles.append(model_handle)
        all_labels.append(model_label)



# Ensure y-limit starts at 0
max_overall_mean = max(max(means) for means in combined_means.values() if means) if combined_means else 0
ax.set_ylim(bottom=0, top=max(10, max_overall_mean * 1.15))

ax.set_ylabel('Percentage of Predictions (%)', fontsize=16)
ax.set_xlabel('Predicted as', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax.set_axisbelow(True)


dataset_handles = [h for h, l in zip(all_handles, all_labels) if l in dataset_display_names.values()]
model_handles = [h for h, l in zip(all_handles, all_labels) if l in model_display_names.values()]
dataset_labels = [l for l in all_labels if l in dataset_display_names.values()]
model_labels = [l for l in all_labels if l in model_display_names.values()]

# Add a third legend entry for the "Correct Class" highlight
highlight_handle = patches.Patch(facecolor='white', edgecolor='black', linewidth=3, label="Correct Class")
highlight_label = "Correct Class (Black Border)"

all_handles_combined = dataset_handles + model_handles + [highlight_handle]
all_labels_combined = dataset_labels + model_labels + [highlight_label]

# Use a single, consolidated legend placed outside the plot area, with multiple columns for compactness
legend = ax.legend(all_handles_combined, all_labels_combined,
                   title="Legend",
                   title_fontsize='14', fontsize='14',
                   loc='upper left',
                   bbox_to_anchor=(0.95, 1.0),  # Position outside the plot on the right
                   ncol=1,  # 1 column for readability
                   frameon=True)

# Adjust layout to make space for the legend
plt.subplots_adjust(right=0.8)  # Reserve 20% of the figure for the legend

plt.tight_layout(rect=(0, 0, 0.95, 1))  # Final adjustment

safe_output_name = f"{'_vs_'.join(METHODS)}_combined_bar_plot.png"
plt.savefig(f"{safe_output_name}")
plt.close(fig)

print(f"\nProcessing complete. Combined plot saved to {safe_output_name}")