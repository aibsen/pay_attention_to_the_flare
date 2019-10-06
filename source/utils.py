import pandas as pd
import numpy as np
import pickle
import os
import csv
import torch
import argparse

extragalactic = [90,67,52,42,62,95,15,64,88]
extragalactic_names = [
    "SNIa",
    "SNIa91bg",
    "SNIax",
    "SNII",
    "SNIbc",
    "SLSN-I",
    "TDE",
    "KN",
    "AGN"]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_meta_data(metadata_file):
        metadata = pd.read_csv(metadata_file, sep=",", names=["object_id","ra","decl",
        "ddf_bool","hostgal_specz","hostgal_photoz","hostgal_photoz_err","distmod","mwebv",
        "target","true_target","true_submodel","true_z","true_distmod","true_lensdmu",
        "true_vpec","true_rv","true_av","true_peakmjd","libid_cadence","tflux_u","tflux_g",
        "tflux_r","tflux_i","tflux_z","tflux_y"], skiprows=1)
        return metadata


def load_csv_data(data_file, metadata_file):
        data = pd.read_csv(data_file, sep=",", names=["object_id","mjd","passband","flux","flux_err","detected_bool"], skiprows=1)
        ids = data['object_id']
        ids = ids.drop_duplicates().values
        metadata = pd.read_csv(metadata_file, sep=",", names=["object_id","ra","decl",
        "ddf_bool","hostgal_specz","hostgal_photoz","hostgal_photoz_err","distmod","mwebv",
        "target","true_target","true_submodel","true_z","true_distmod","true_lensdmu",
        "true_vpec","true_rv","true_av","true_peakmjd","libid_cadence","tflux_u","tflux_g",
        "tflux_r","tflux_i","tflux_z","tflux_y"], skiprows=1)
        return [data, ids, metadata]


def retag(targets):
    retagged = targets
    for i,c in enumerate(extragalactic):
        retagged.loc[retagged==c] = i
    return retagged

def save_to_stats_pkl_file(experiment_log_filepath, filename, stats_dict):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "wb") as file_writer:
        pickle.dump(stats_dict, file_writer)


def load_from_stats_pkl_file(experiment_log_filepath, filename):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "rb") as file_reader:
        stats = pickle.load(file_reader)

    return stats

def save_classification_results(experiment_log_dir, filename, results, tags):
        results_filename = os.path.join(experiment_log_dir, filename)
        classes = np.arange(9)
        results_df = pd.DataFrame(results.cpu().numpy(),columns=classes)
        results_df['true_tags'] = pd.Series(tags.cpu().numpy())
        p = torch.argmax(results, dim=1)
        results_df['predicted_tags'] = pd.Series(p.cpu().numpy())
        results_df.to_csv(results_filename,sep=",",index=False)


def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch, save_full_dict=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'w' if ((current_epoch == 0) or (save_full_dict == True)) else 'a'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if current_epoch == 0:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename


def load_statistics(experiment_log_dir, filename):
    """
    Loads a statistics csv file into a dictionary
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file to load
    :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
     particular header are converted into values of a key in a list format.
    """
    summary_filename = os.path.join(experiment_log_dir, filename)

    with open(summary_filename, 'r+') as f:
        lines = f.readlines()

    keys = lines[0].split(",")
    stats = {key: [] for key in keys}
    for line in lines[1:]:
        values = line.split(",")
        for idx, value in enumerate(values):
            stats[keys[idx]].append(value)

    return stats