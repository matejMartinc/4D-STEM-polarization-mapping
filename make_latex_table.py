import pandas as pd
import numpy as np
import os



models = [
    'conv_regression', 'conv_classification', 'conv_just_contrastive',
    'vgg_regression', 'vgg_classification', 'vgg_just_contrastive',
    'res_net_regression', 'res_net_classification', 'res_net_just_contrastive', 'pca_20'
]

seeds = [888, 1212, 3000, 5000, 7000]


datasets = [
    'test_results',
    'LU_4DSTEM_20nm_NoTDS',
    'KNN_Right_Mid_20nm_24mrad16383',
    'RU_KNN_Simpl_20nm_24mrad_NoTDS16383',
    'LD_KNN_Simpl_20nm_24mrad_NoTDS',
    'KNN_Simpl_Right_Down_20nm',
    'LU_4DSTEM_50nm_NoTD',
    'RD_KNN_Simpl_50nm_24mrad_NoTDS16383',
    'KNNsc_40MX_0V_a',
    'KNNsc_40MX_0V_b',
    'KNNsc_40MX_0V_c',
    'KNN_Janina_a',
    'KNNsc_tripod_tiff',
]

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

folder = "results_final_filtering_preprocessing"

pca_acc = {
     "results_final": [0.9501830942935612, 0.95155630149527, 0.9487335978028685, 0.9537686908758011, 0.952014037229173],
     "results_final_preprocessing": [0.3567287152883735, 0.35993286542569425, 0.362221544095209, 0.3581019224900824, 0.3645865120537077],
     "results_final_filtering_preprocessing": [0.3838276836158192, 0.3908898305084746, 0.391066384180791, 0.3822387005649718, 0.390183615819209]
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
    'test_results': 'Synthetic test',
    'KNN_Janina_a_45_degROT': '45_degROT',
    'KNN_Janina_a_90_degROT': '90_degROT',
    'KNN_Janina_a_180_degROT': '180_degROT',
    'KNNsc_40MX_minus20V_a': '40MX -20V',
    'KNN_Experimental/KNNsc_40MX_20V_a': 'Exp 40MX 20V',
    'KNN_Scr_12b_x100016383': 'KNN Scr 12b',
}

RM_CLASS = 'RM'
RD_CLASS = 'RD'

# Define which datasets allow multiple "correct" answers
multi_label_datasets = {
    'KNN_Janina_a': [RM_CLASS, RD_CLASS],
    'KNNsc_tripod_tiff': [RM_CLASS, RD_CLASS]
}


# --- Calculation Logic ---

results_table = {}
folder = "results_final_filtering_preprocessing"

for dataset in datasets:
    results_table[dataset] = {}
    true_majority_class = None

    # Check if this specific dataset has flexible ground truths
    allowed_classes = multi_label_datasets.get(dataset, None)

    for model in models:
        accuracies = []
        all_preds_for_model = []

        for seed in seeds:
            filename = f"{folder}/{model}_{seed}_{dataset}.tsv"

            # Handle PCA exception
            if filename == folder + '/pca_20_' + str(seed) + '_test_results.tsv':
                accuracies = pca_acc[folder]
            else:
                df = pd.read_csv(filename, sep='\t')

                if allowed_classes:
                    # Correct if prediction is in the list of allowed classes
                    correct = df['pred'].isin(allowed_classes).sum()
                else:
                    # Standard check
                    correct = (df['pred'] == df['true']).sum()

                total = len(df)
                accuracies.append(correct / total if total > 0 else 0)
                all_preds_for_model.extend(df['pred'].tolist())

                if true_majority_class is None and len(df) > 0:
                    true_majority_class = df['true'].mode()[0]

        # Aggregate Stats
        if accuracies:
            results_table[dataset][model] = {
                'mean': np.mean(accuracies) * 100,
                'std': np.std(accuracies) * 100,
                'pred_major': pd.Series(all_preds_for_model).mode()[0] if all_preds_for_model else -1
            }
        else:
            results_table[dataset][model] = {'mean': -1, 'std': -1, 'pred_major': -1}

    results_table[dataset]['TRUE_MAJORITY'] = true_majority_class if true_majority_class is not None else -1

print("Processing complete. Generating LaTeX...")

# --- LaTeX Generation ---

latex_str = []
latex_str.append(r"\begin{table*}[ht]")
latex_str.append(r"\centering")
latex_str.append(r"\resizebox{\textwidth}{!}{")
latex_str.append(r"\setlength{\tabcolsep}{2pt}")
latex_str.append(r"\begin{tabular}{l" + "c" * len(models) + "}")
latex_str.append(r"\hline")

# Header
header = "Dataset"
for model in models:
    header += f" & {model_display_names[model]}"
header += r" \\"
latex_str.append(header)
latex_str.append(r"\hline")

for dataset in datasets:
    display_name = dataset_display_names.get(dataset, dataset)
    true_major = results_table[dataset]['TRUE_MAJORITY']

    # 1. FIND MAX ACCURACY FOR THIS DATASET ROW
    max_acc = -1
    for model in models:
        acc = results_table[dataset][model]['mean']
        if acc > max_acc:
            max_acc = acc

    # --- Row 1: Accuracy ---
    row1 = f"\\textbf{{{display_name}}}"
    for model in models:
        data = results_table[dataset][model]
        mean_acc = data['mean']
        std_acc = data['std']

        if mean_acc == -1 and std_acc == -1:
            cell_content = "N/A"
        else:
            # Check for max accuracy (using tolerance for float comparison)
            is_max = abs(mean_acc - max_acc) < 1e-9 and max_acc != -1

            # Format the numbers
            mean_str = f"{mean_acc:.1f}"
            std_str = f"{std_acc:.1f}"

            # Apply bolding if it's the max accuracy
            if is_max:
                # Use \mathbf{} to bold the numbers inside math mode
                cell_content = f"$\\mathbf{{{mean_str}}} \\pm \\mathbf{{{std_str}}}$"
            else:
                # Standard math mode formatting
                cell_content = f"${mean_str} \\pm {std_str}$"

        row1 += f" & {cell_content}"
    row1 += r" \\"
    latex_str.append(row1)

    row2 = f"\\textit{{Maj. Class}}"
    for model in models:
        data = results_table[dataset][model]
        pred_major = data['pred_major']

        if pred_major == -1:
            row2 += " & -"
        else:
            # Check if prediction is valid for this dataset
            allowed = multi_label_datasets.get(dataset, [true_major])

            if pred_major in allowed:
                color_cmd = r"\textcolor{darkgreen}"
            else:
                color_cmd = r"\textcolor{red}"

            row2 += f" & {color_cmd}{{{pred_major}}}"
    row2 += r" \\"
    latex_str.append(row2)
    latex_str.append(r"\hline")


latex_str.append(r"\end{tabular}")
latex_str.append(r"}")
latex_str.append(
    r"\caption{Accuracy (top row) and Majority Class Prediction (bottom row). The highest accuracy in each row is \textbf{bolded}. \textcolor{darkgreen}{Green} indicates the predicted majority matches the ground truth; \textcolor{red}{Red} indicates a mismatch.}")
latex_str.append(r"\label{tab:results_detailed}")
latex_str.append(r"\end{table*}")

final_latex = "\n".join(latex_str)

print("\n" + "=" * 30)
print("LATEX OUTPUT (With Bolding)")
print("=" * 30 + "\n")
print(final_latex)

with open("results_table_detailed_bolded.tex", "w") as f:
    f.write(final_latex)