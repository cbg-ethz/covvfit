## scritps to generate the simulation ##

import yaml
import numpy as np
import pandas as pd
import os
import sys 
import glob

def load_yaml(yaml_file):
    """
    Load content from a YAML file.

    Args:
    - yaml_file (str): Path to the YAML file.

    Returns:
    - dict: Content of the YAML file.
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_all_yamls(directory_pattern):
    """
    Load content from all YAML files matching the directory pattern.

    Args:
    - directory_pattern (str): Directory pattern (with wildcards) to search for YAML files.

    Returns:
    - dict of dict: Dictionary with filenames as keys and content of each YAML file as values.
    """
    yaml_files = glob.glob(directory_pattern)
    data = {}
    for yaml_file in yaml_files:
        data[yaml_file] = load_yaml(yaml_file)
    return data


def generate_variant_definition_matrix(all_data):
    all_mutations = set()
    all_variants = []
    for file_name, content in all_data.items():
        all_mutations.update(content['mut'].keys())
        all_variants.append(content['variant']['short'])

    df = pd.DataFrame(0, index=sorted(all_mutations, key=int), columns=all_variants)
    for file_name, content in all_data.items():
        variant = content['variant']['short']
        mutations = content['mut'].keys()
        for mutation in mutations:
            df.at[mutation, variant] = 1
    return df

def compute_mutation_fractions(df, bb1, variant_names):
    """
    Compute mutation fractions.

    Args:
    - df (pd.DataFrame): Variant definition matrix.
    - bb1 (np.ndarray): bb1 matrix.

    Returns:
    - pd.DataFrame: Computed mutation fractions.
    """
    return df[variant_names].dot(bb1)


def generate_coverage_matrix(rng, mu=1000, n=5, M=None, T=None):
    p = n / (mu + n)
    return rng.negative_binomial(n, p, size=(M, T))


def sample_observed_counts(rng, yy1, overdisp, mut_rate, array):
    yy2 = yy1.values
    yy2 = (yy2 + mut_rate) / (1 + 2 * mut_rate)
    alpha = yy2 * overdisp
    beta = (1 - yy2) * overdisp
    result = np.zeros_like(yy1, dtype=int)
    for m in range(yy1.shape[0]):
        for t in range(yy1.shape[1]):
            p = rng.beta(alpha[m, t], beta[m, t])
            result[m, t] = rng.binomial(array[m, t], p)
    return result


def melt_dataframe(yy1):
    yy_tmp = yy1.T.copy()
    yy_tmp.index.name = 'date'
    return yy_tmp.reset_index().melt(id_vars='date', var_name='pos', value_name='value')


def add_variant_definitions_to_melted(melted_df, variants_df):
    for variant in variants_df.columns:
        melted_df[variant] = melted_df['pos'].map(variants_df[variant]).fillna(0).astype(int)
    melted_df['date'] = pd.to_datetime(melted_df['date'], unit='D', origin='1970-01-01')
    return melted_df


def generate_xx1_bb1(points, a1, m1):
    xx1 = np.arange(points)
    bb1 = softmax_1(xx1, a1, m1)
    return xx1, bb1

def softmax_1(x, rates, midpoints):
    un_norm = np.exp(rates[:, np.newaxis] * (x - midpoints[:, np.newaxis]))
    return un_norm / (un_norm.sum(axis=0))

def logit_inv(x):
    return np.exp(x)/(1+np.exp(x))
    

def generate_final_df(rng_seed, directory, variant_names, points, a1, m1, mu, n, dropout_prob, mut_rate, overdisp):
    # Instantiate the random number generator
    rng = np.random.default_rng(rng_seed)

    # Load YAML data and generate the variant definition matrix
    all_data = load_all_yamls(directory)
    variant_def_matrix = generate_variant_definition_matrix(all_data)

    # Subset only the columns for variant_names, and keep only informative rows:
    variant_def_matrix = variant_def_matrix[variant_names]
    # Remove rows that are all 0s or all 1s
    row_sums = variant_def_matrix.sum(axis=1)
    variant_def_matrix = variant_def_matrix[(row_sums != 0) & (row_sums != len(variant_names))]
    
    # Generate xx1 and bb1
    xx1, bb1 = generate_xx1_bb1(points, a1, m1)

    # Use variant definitions to create smooth time series of mutation fractions
    yy1 = compute_mutation_fractions(variant_def_matrix, bb1, variant_names)

    # Generate coverage matrix
    M, T = yy1.shape
    coverage_matrix = generate_coverage_matrix(rng, M=M, T=T, mu=mu, n=n)
    coverage_matrix[coverage_matrix<5] = 0 

    # Randomly set some read depth to zero
    # Flatten the coverage matrix to select exact number of dropouts
    total_elements = coverage_matrix.size
    num_dropouts = int(np.round(dropout_prob * total_elements))
    flat_indices = rng.permutation(total_elements)[:num_dropouts]
    # Set selected indices to zero
    coverage_matrix_flat = coverage_matrix.flatten()
    coverage_matrix_flat[flat_indices] = 0
    coverage_matrix = coverage_matrix_flat.reshape(coverage_matrix.shape)

    # Sample observed counts
    observed_counts = sample_observed_counts(rng, yy1, overdisp=overdisp, mut_rate=mut_rate, array=coverage_matrix)

    # Melt the observed counts dataframe
    melted_df = melt_dataframe(pd.DataFrame(observed_counts / coverage_matrix, index=yy1.index, columns=yy1.columns))

    # Add variant definitions to the melted dataframe
    final_df = add_variant_definitions_to_melted(melted_df, variant_def_matrix[variant_names])

    # Drop rows with NaN values in the 'value' column
    final_df = final_df.dropna(subset=['value'])

    # Drop rows where all columns from variant_names have 0
    condition = final_df[variant_names].sum(axis=1) != 0
    final_df = final_df[condition]
    
    return final_df, xx1, bb1, yy1, observed_counts, coverage_matrix

