from preprocessing import KNNDataModule, create_dataframe
import lightning as pl
import torch
import joblib
from model import LitModel
from sklearn.model_selection import train_test_split
from lightning.pytorch import seed_everything
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict, Counter
import os
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def find_most_similar_key(dictionary, target_vector, pr=True):
    max_similarity = -1.0  # Cosine similarity ranges from -1 to 1
    most_similar_key = None

    # Ensure target_vector is 2D
    target_vector = target_vector.reshape(1, -1)

    for key, vector in dictionary.items():
        vector = vector.reshape(1, -1)
        similarity = cosine_similarity(vector, target_vector)[0][0]
        if pr:
            print(key, similarity)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_key = key

    return most_similar_key



def prepare_data(df, image_size, backbone, train_set_chunk=0.8):
    if train_set_chunk == 1:
        df_train = df
        df_test = df
        df_val = df
    else:
        test_size = 1 - train_set_chunk
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
        df_test, df_val = train_test_split(df_test, test_size=0.5, random_state=42)

    dataframes = {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }

    dm = KNNDataModule(dataframes=dataframes, image_size=image_size, batch_size=128, backbone=backbone)
    dm.prepare_data()
    dm.setup()
    return dm


if __name__ == "__main__":
    device = torch.device('cuda')
    configs = ['conv_classification', 'conv_regression', 'conv_just_contrastive'
               'vgg_regression', 'vgg_classification', 'vgg_just_contrastive',
               'res_net_regression', 'res_net_classification', 'res_net_just_contrastive']

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

    for config in configs:
        if 'res_net' in config or 'vgg' in config:
            backbone = True
        else:
            backbone = False
        if 'res_net' in config:
            encoder_name = 'res_net'
        elif 'vgg' in config:
            encoder_name = 'vgg'
        else:
            encoder_name = 'custom'
        all_results = defaultdict(list)

        for seed in [888, 1212, 3000, 5000, 7000]:
            print('\n\n----------------------------------------------------------------------')
            print('Running seed', seed)
            print('Using backbone', config)
            print('----------------------------------------------------------------------\n\n')
            seed_everything(seed, workers=True)
            path = 'data/KNN_New_Simulations'
            training_files = joblib.load("training_files_mean_largest_magnitude.pth")
            # Set training files to None if you do not want to filter train set
            # training_files = None
            results_prependix = 'results_final_filtering_preprocessing/'
            if not os.path.exists(results_prependix):
                os.makedirs(results_prependix)

            image_size = 64
            df_unsupervised, _, _ = create_dataframe(path, training_files=training_files)
            dm = prepare_data(df_unsupervised, image_size, backbone)

            # get avg representations for test
            experimental_paths = [['data/KNN_Experimental/KNN_Janina_a'],
                                  ['data/KNN_Experimental/KNNsc_tripod_tiff'],
                                   ['data/KNN_Experimental/KNNsc_40MX_0V_a'],
                                   ['data/KNN_Experimental/KNNsc_40MX_0V_b'],
                                   ['data/KNN_Experimental/KNNsc_40MX_0V_c'],
                                   ['data/LU_4DSTEM/LU_4DSTEM_20nm_NoTDS'],
                                   ['data/LU_4DSTEM/LU_4DSTEM_50nm_NoTD'],
                                   ['data/KNN_Right_Mid_20nm_24mrad16383'],
                                   ['data/RD_KNN_Simpl_50nm_24mrad_NoTDS16383'],
                                   ['data/RU_KNN_Simpl_20nm_24mrad_NoTDS16383'],
                                   ['data/LD_KNN_Simpl_20nm_24mrad_NoTDS'],
                                   ['data/KNN_Simpl_Right_Down_20nm'],
                                   ['data/KNN_Experimental/Simulated_anomaly'],
                                  ]
            labels = ['RM', 'RM', 'RM', 'RM', 'RM', 'LU', 'LU', 'RM', 'RD', 'RU', 'LD', 'RD', 'RD']
            output_paths = [results_prependix + config + '_' + str(seed) + '_KNN_Janina_a.tsv',
                             results_prependix + config + '_' + str(seed) + '_KNNsc_tripod_tiff.tsv',
                             results_prependix + config + '_' + str(seed) + '_KNNsc_40MX_0V_a.tsv',
                             results_prependix + config + '_' + str(seed) + '_KNNsc_40MX_0V_b.tsv',
                             results_prependix + config + '_' + str(seed) + '_KNNsc_40MX_0V_c.tsv',
                             results_prependix + config + '_' + str(seed) + '_LU_4DSTEM_20nm_NoTDS.tsv',
                             results_prependix + config + '_' + str(seed) + '_LU_4DSTEM_50nm_NoTD.tsv',
                             results_prependix + config + '_' + str(seed) + '_KNN_Right_Mid_20nm_24mrad16383.tsv',
                             results_prependix + config + '_' + str(seed) + '_RD_KNN_Simpl_50nm_24mrad_NoTDS16383.tsv',
                             results_prependix + config + '_' + str(seed) + '_RU_KNN_Simpl_20nm_24mrad_NoTDS16383.tsv',
                             results_prependix + config + '_' + str(seed) + '_LD_KNN_Simpl_20nm_24mrad_NoTDS.tsv',
                             results_prependix + config + '_' + str(seed) + '_KNN_Simpl_Right_Down_20nm.tsv',
                             results_prependix + config + '_' + str(seed) + '_Simulated_anomaly.tsv',
                           ]

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=True,
                mode="min"
            )

            # Save only the best model based on val_loss
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"checkpoints/{config}/{seed}",
                filename="best-checkpoint",
                save_top_k=1,
                mode="min"
            )

            callbacks_list = [early_stop_callback, checkpoint_callback]

            if 'just_contrastive' in config:
                # Initialize a trainer
                model = LitModel((1, image_size, image_size), 8, contrastive_loss=True, regression_training=False,
                                 seed=seed, backbone=backbone, encoder_name=config, output_folder=results_prependix)
                model.to(device)
                trainer = pl.Trainer(max_epochs=20,
                                     precision="16-mixed" if torch.cuda.is_available() else "32-true",
                                     devices=1,
                                     accelerator="gpu",
                                     callbacks=callbacks_list,
                                     )

                # Train the model ⚡🚅⚡
                trainer.fit(model, dm)

                print(f"Loading best model from: {checkpoint_callback.best_model_path}")
                model = LitModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    input_shape=(1, image_size, image_size),
                    num_classes=8,
                    contrastive_loss=True,
                    regression_training=False,
                    seed=seed,
                    backbone=backbone,
                    encoder_name=config,
                    output_folder=results_prependix
                )
                model.to(device)
                model.eval()

                val_loader = dm.val_dataloader()
                avg_representations = {}
                for batch in val_loader:
                    _ , x, y = batch
                    batch_representation = model(x.to(device))
                    batch_representation = batch_representation.view(batch_representation.shape[0], -1)

                    for idx, example in enumerate(batch_representation):
                        if y[idx].item() not in avg_representations:
                            avg_representations[y[idx].item()] = torch.zeros((example.shape[0]))
                        avg_representations[y[idx].item()] += example.cpu().detach()

                print('Calculated avg. representation on synthetic dataset')

                test_loader = dm.test_dataloader()
                results_test = []
                for batch in test_loader:
                    paths, x, y = batch
                    batch_representation = model(x.to(device))
                    batch_representation = batch_representation.view(batch_representation.shape[0], -1)
                    for path, label, example in zip(paths, y, batch_representation):
                        example = example.cpu().detach()
                        most_similar = find_most_similar_key(avg_representations, example, pr=False)
                        results_test.append((path, idx2label[most_similar], idx2label[label.item()]))
                df = pd.DataFrame(results_test, columns=['df_index', 'pred', 'true'])
                df.to_csv(results_prependix + "/" + config + '_' + str(seed) + "_test_results.tsv", index=False, sep='\t')

                print('Calculated scores on synthetic test set')

                for experimental_path, label, output_path in zip(experimental_paths, labels, output_paths):
                    df_unsupervised, _, _ = create_dataframe(experimental_path)
                    dm = prepare_data(df_unsupervised, image_size, backbone, train_set_chunk=1)
                    train_loader = dm.train_dataloader()
                    avg_test_vector = torch.zeros((128))
                    results = []
                    for batch in train_loader:
                        f_paths, x, y = batch
                        batch_representation = model(x.to(device))
                        batch_representation = batch_representation.view(batch_representation.shape[0], -1)
                        for idx, example in enumerate(batch_representation):
                            example = example.cpu().detach()
                            avg_test_vector += example
                            most_similar = find_most_similar_key(avg_representations, example, pr=False)
                            f_path = f_paths[idx].split('.')[0][-5:]
                            results.append((f_path, idx2label[most_similar], label))
                    df = pd.DataFrame(results, columns=['df_index', 'pred', 'true'])
                    df.to_csv(output_path, index=False, sep='\t')
                    most_similar = find_most_similar_key(avg_representations, avg_test_vector)
                    print('Most similar for seed', seed, experimental_path[0], ":", idx2label[most_similar])
                    print('Counter preds', Counter(df['pred'].tolist()))
                    print(max(set(df['pred'].tolist()), key=df['pred'].tolist().count))
                    all_results[experimental_path[0]].append(max(set(df['pred'].tolist()), key=df['pred'].tolist().count))

            elif 'regression' in config:
                # Initialize a trainer
                model = LitModel((1, image_size, image_size), 8, contrastive_loss=False, regression_training=True, seed=seed, backbone=backbone, encoder_name=config, output_folder=results_prependix)
                model.to(device)
                trainer = pl.Trainer(max_epochs=20,
                                     precision="16-mixed" if torch.cuda.is_available() else "32-true",
                                     devices=1,
                                     accelerator="gpu",
                                     callbacks=callbacks_list,
                                     )

                # Train the model ⚡🚅⚡
                trainer.fit(model, dm)
                trainer.test(dataloaders=dm.test_dataloader(), ckpt_path=checkpoint_callback.best_model_path)

                # --- LOAD BEST MODEL ---
                print(f"Loading best model from: {checkpoint_callback.best_model_path}")
                model = LitModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    input_shape=(1, image_size, image_size),
                    num_classes=8,
                    contrastive_loss=False,
                    regression_training=True,
                    seed=seed,
                    backbone=backbone,
                    encoder_name=config,
                    output_folder=results_prependix
                )
                model.to(device)
                model.eval()

                print('Majority classification fro regression')

                for experimental_path, label, output_path in zip(experimental_paths, labels, output_paths):
                    df_unsupervised, _, _ = create_dataframe(experimental_path)
                    dm = prepare_data(df_unsupervised, image_size, backbone, train_set_chunk=1)
                    train_loader = dm.train_dataloader()
                    results = []
                    counter = defaultdict
                    all_preds = []
                    for batch in train_loader:
                        f_paths, x, y = batch
                        logits = model(x.to(device))
                        preds = torch.argmax(logits, dim=1)
                        for idx, example in enumerate(preds):
                            example_label = idx2label[example.item()]
                            f_path = f_paths[idx].split('.')[0][-5:]
                            results.append((f_path, example_label, label))
                            all_preds.append(example_label)
                    df = pd.DataFrame(results, columns=['df_index', 'pred', 'true'])
                    df.to_csv(output_path, index=False, sep='\t')
                    most_similar = Counter(all_preds).most_common(1)[0][0]
                    print('Most similar for config and seed', config, seed, experimental_path[0], ":", most_similar)
                    all_results[experimental_path[0]].append(most_similar)
            else:
                # Initialize a trainer
                model = LitModel((1, image_size, image_size), 8, contrastive_loss=False,
                                 regression_training=False, seed=seed, backbone=backbone, encoder_name=config, output_folder=results_prependix)
                model.to(device)
                trainer = pl.Trainer(max_epochs=20,
                                     precision="16-mixed" if torch.cuda.is_available() else "32-true",
                                     devices=1,
                                     accelerator="gpu",
                                     callbacks=callbacks_list,
                                     )

                # Train the model ⚡🚅⚡
                trainer.fit(model, dm)
                trainer.test(dataloaders=dm.test_dataloader(), ckpt_path=checkpoint_callback.best_model_path)

                # --- LOAD BEST MODEL ---
                print(f"Loading best model from: {checkpoint_callback.best_model_path}")
                # We load the best weights. We pass the original arguments to ensure correct architecture initialization.
                model = LitModel.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    input_shape=(1, image_size, image_size),
                    num_classes=8,
                    contrastive_loss=False,
                    regression_training=False,
                    seed=seed,
                    backbone=backbone,
                    encoder_name=config,
                    output_folder=results_prependix,
                )
                model.to(device)
                model.eval()
                print('Majority classification')

                for experimental_path, label, output_path in zip(experimental_paths, labels, output_paths):
                    df_unsupervised, _, _ = create_dataframe(experimental_path)
                    dm = prepare_data(df_unsupervised, image_size, backbone, train_set_chunk=1)
                    train_loader = dm.train_dataloader()
                    results = []
                    counter = defaultdict
                    all_preds = []
                    for batch in train_loader:
                        f_paths, x, y = batch
                        logits = model(x.to(device))
                        preds = torch.argmax(logits, dim=1)
                        for idx, example in enumerate(preds):
                            example_label = idx2label[example.item()]
                            f_path = f_paths[idx].split('.')[0][-5:]
                            results.append((f_path, example_label, label))
                            all_preds.append(example_label)
                    df = pd.DataFrame(results, columns=['df_index', 'pred', 'true'])
                    df.to_csv(output_path, index=False, sep='\t')
                    most_similar = Counter(all_preds).most_common(1)[0][0]
                    print('Most similar for config and seed', config, seed, experimental_path[0], ":", most_similar)
                    all_results[experimental_path[0]].append(most_similar)

        print("Results per config", config, all_results)




