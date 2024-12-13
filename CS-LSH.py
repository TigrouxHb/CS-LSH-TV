# Author Henri Botermans
# Friday 13 December 2024

# Import
import json
import pandas as pd
import numpy as np
import re
from sklearn.utils import resample
import collections
import itertools
from sklearn.cluster import AgglomerativeClustering
import random
import time
import matplotlib.pyplot as plt
import logging

#                 CONFIGURATION & PARAMETERS              #

JSON_PATH = "/Users/henribotermans/Library/Mobile Documents/com~apple~CloudDocs/Rotterdam/Period 2/Computer Science/Papers/Final/Data/TVs-all-merged.json"  # Adjust path to your JSON dataset
N_BOOTSTRAPS = 5 # Write in the assignment
QGRAM_SIZE = 3 # Personnel choice after tuning and reference from the paper
N_HASHES = 300 # Personnel choice after tuning
LSH_THRESHOLD_VALUES = [0.05 * i for i in range(1, 20)]
RANDOM_SEED = 177 # Personnel choice
random.seed(RANDOM_SEED)
CLUSTERING_THRESHOLD_STEPS = 20
cluster_threshold_cache = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


#                  FUNCTIONS                        #

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_data_from_json(data):
    items = []
    for model_id, product_list in data.items():
        for item in product_list:
            title = item.get('title', '').strip()
            shop = item.get('shop', '').strip().lower()
            brand = item.get('featuresMap', {}).get('Brand', '').lower().strip()
            features = item.get('featuresMap', {})
            items.append((model_id, title, shop, brand, features))
    df = pd.DataFrame(items, columns=['modelID', 'title', 'shop', 'brand', 'features'])
    return df

# Clean dataset
def clean_text(text):
    t = text.lower()
    t = t.replace("\"", "inch").replace("inches", "inch").replace("'", "inch")
    t = t.replace("hertz", "hz")

    remove_words = ["best buy", "newegg.com", "thenerds.net", "amazon"]
    for w in remove_words:
        t = t.replace(w, "")
    t = re.sub(r'[^a-z0-9\s]', '', t)  # Remove non-alphanumeric characters
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def clean_data_frame(df):
    df['title'] = df['title'].apply(clean_text)
    df['brand'] = df['brand'].apply(clean_text)
    return df


def bootstrap_sample(df, boot_i):
    random_state = RANDOM_SEED + boot_i
    train_data = resample(df, replace=True, n_samples=len(df), random_state=random_state)
    id_set_train = set(train_data.index)
    test_data = df.loc[~df.index.isin(id_set_train)]
    return train_data, test_data


def extract_model_words_from_title(title):
    tokens = title.split()
    model_words = []     # Consider all attributes, try extracting numeric tokens
    for tok in tokens:
        if re.search('[0-9]', tok) and re.search('[a-z]', tok):
            model_words.append(tok)
        elif tok.isdigit():
            model_words.append(tok)
    return model_words


def extract_numeric_model_words_from_attributes(features):
    model_words = []
    # Consider all attributes, try extracting numeric tokens
    for k, v in features.items():
        v_norm = clean_text(v)
        nums = re.findall(r'\d+(\.\d+)?[a-z]*', v_norm)
        for numw in nums:
            val = re.sub('[^0-9\.]', '', numw)
            if val != '':
                model_words.append(val)
    return model_words


def build_vocabulary(df):
    vocab = set()
    for i, row in df.iterrows():
        title_mw = extract_model_words_from_title(row['title'])
        attr_mw = extract_numeric_model_words_from_attributes(row['features'])
        mw = set(title_mw + attr_mw)
        vocab.update(mw)
    vocab = list(vocab)
    return vocab


def create_binary_matrix(df, vocab):
    vocab_index = {w: i for i, w in enumerate(vocab)}
    n_products = len(df)
    n_vocab = len(vocab)
    matrix = np.zeros((n_vocab, n_products), dtype=np.int8)
    for j, row in enumerate(df.itertuples()):
        title_mw = extract_model_words_from_title(row.title)
        attr_mw = extract_numeric_model_words_from_attributes(row.features)
        mw_set = set(title_mw + attr_mw)
        for mw in mw_set:
            if mw in vocab_index:
                matrix[vocab_index[mw], j] = 1
    return matrix


def generate_hash_functions(n_hashes, max_val):
    p = 2147483647 # prime number
    hash_funcs = []
    for i in range(n_hashes):
        a = random.randint(1, p - 1)
        b = random.randint(0, p - 1)
        hash_funcs.append((a, b, p))
    return hash_funcs


def compute_minhash_signatures(boolean_matrix, n_hashes):
    n_rows, n_cols = boolean_matrix.shape
    hash_functions = generate_hash_functions(n_hashes, n_rows)
    signatures = np.full((n_hashes, n_cols), np.inf)

    for row in range(n_rows):
        row_has_value = np.where(boolean_matrix[row, :] == 1)[0]
        if len(row_has_value) == 0:
            continue
        for h_i, (a, b, p) in enumerate(hash_functions):          # Compute hash value for the current row
            h_val = (a * row + b) % p
            current_sign = signatures[h_i, row_has_value]
            signatures[h_i, row_has_value] = np.minimum(current_sign, h_val)
    return signatures


def find_band_row_config(n_hashes, t):
    best_diff = float('inf')
    best_b, best_r = None, None
    for r_candidate in range(1, n_hashes + 1):
        if n_hashes % r_candidate == 0:
            b_candidate = n_hashes // r_candidate        # Approximate threshold based on the banding technique formula
            approx_t = (1 / b_candidate) ** (1 / r_candidate)
            diff = abs(approx_t - t)
            if diff < best_diff:
                best_diff = diff
                best_b = b_candidate
                best_r = r_candidate
    return best_b, best_r


def locality_sensitive_hashing(signatures, threshold):
    n_hashes, n_products = signatures.shape
    b, r = find_band_row_config(n_hashes, threshold)    # Determine the optimal number of bands and rows per band based on the threshold
    logging.info(f"Using LSH with b={b}, r={r}, t={threshold}")

    band_hash = collections.defaultdict(set)
    for band_i in range(b):
        start = band_i * r
        end = start + r
        band_data = signatures[start:end, :]
        for c in range(n_products):
            band_key = tuple(band_data[:, c])
            band_hash[(band_i, band_key)].add(c)


    candidate_pairs = set()
    for s in band_hash.values():
        if len(s) > 1:
            for pair in itertools.combinations(s, 2):
                candidate_pairs.add(tuple(sorted(pair)))
    return candidate_pairs, b, r


def count_duplicates(df):
    model_groups = df.groupby('modelID').size()
    dups = 0
    for c in model_groups:
        if c > 1:
            dups += (c * (c - 1)) // 2
    return dups


def qgram_similarity(s1, s2, q=3): # personnel choice
    def qgrams(st):
        return {st[i:i + q] for i in range(len(st) - q + 1)} if len(st) >= q else {st}
    Q1 = qgrams(s1)
    Q2 = qgrams(s2)
    inter = len(Q1.intersection(Q2))
    uni = len(Q1.union(Q2))
    return inter / uni if uni > 0 else 0


def get_attribute_model_words(df, idx):
    # Extract model words for attributes for given product index
    row = df.iloc[idx]
    return set(extract_numeric_model_words_from_attributes(row['features']))


def jaccard_similarity(set1, set2):
    inter = len(set1.intersection(set2))
    uni = len(set1.union(set2))
    return inter / uni if uni > 0 else 0


def product_similarity(df, i, j):

    if df['shop'].iloc[i] == df['shop'].iloc[j]:
        return 0.0    # If both products are from the same shop, similarity is 0 to avoid considering them as duplicates
    if df['brand'].iloc[i] != '' and df['brand'].iloc[j] != '' and df['brand'].iloc[i] != df['brand'].iloc[j]:
        return 0.0

    title_sim = qgram_similarity(df['title'].iloc[i], df['title'].iloc[j], q=QGRAM_SIZE)

    attr_i = get_attribute_model_words(df, i)
    attr_j = get_attribute_model_words(df, j)
    attr_sim = jaccard_similarity(attr_i, attr_j)

    final_sim = 0.7 * title_sim + 0.3 * attr_sim # hardode chose, maybe more tuning
    return final_sim


def build_dissimilarity_matrix(candidate_pairs, df):
    n = len(df)
    dist_matrix = np.full((n, n), 100.0) # Initialize with a large distance
    np.fill_diagonal(dist_matrix, 0)

    for (i, j) in candidate_pairs:
        sim = product_similarity(df, i, j)
        dist = 1 - sim
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
    return dist_matrix


def cluster_products(dist_matrix, threshold):
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='single',
                                         distance_threshold=threshold)
    labels = clustering.fit_predict(dist_matrix)
    clusters = collections.defaultdict(list)
    for idx, l in enumerate(labels):
        clusters[l].append(idx)

    cluster_pairs = set()     # Generate all unique pairs within each cluster
    for cids in clusters.values():
        if len(cids) > 1:
            for pair in itertools.combinations(cids, 2):
                cluster_pairs.add(tuple(sorted(pair)))
    return cluster_pairs


def evaluate_performance(candidate_pairs, cluster_pairs, df):
    total_dups = count_duplicates(df) # Total number of actual duplicate pairs
    duplicates = set()
    model_groups = df.groupby('modelID').indices
    for indices in model_groups.values():
        if len(indices) > 1:
            duplicates.update(itertools.combinations(sorted(indices), 2))

    candidate_set = set(tuple(sorted(p)) for p in candidate_pairs)
    cluster_set = set(tuple(sorted(p)) for p in cluster_pairs)

    found_LSH = len(duplicates.intersection(candidate_set))
    found_cluster = len(duplicates.intersection(cluster_set))

    PQ = found_LSH / (len(candidate_pairs) + 1e-9)
    PC = found_LSH / (total_dups + 1e-9)
    F1 = 2 * (PQ * PC) / (PQ + PC + 1e-9) # Evualtion metric

    precision = found_cluster / (len(cluster_pairs) + 1e-9)
    recall = found_cluster / (total_dups + 1e-9)
    F1_star = 2 * (precision * recall) / (precision + recall + 1e-9)

    N = len(df)
    total_comp = (N * (N - 1)) / 2
    frac_comparison = len(candidate_pairs) / total_comp if total_comp > 0 else 0
    frac_cluster = len(cluster_pairs) / total_comp if total_comp > 0 else 0

    return PQ, PC, F1, precision, recall, F1_star, frac_comparison, frac_cluster


def optimal_threshold(distance_matrix, LSH_pairs, df, threshold_steps=CLUSTERING_THRESHOLD_STEPS):

    cache_key = (id(distance_matrix), frozenset(LSH_pairs))
    if cache_key in cluster_threshold_cache:
        return cluster_threshold_cache[cache_key]

    best_F1_star = 0
    best_threshold = 0.5
    best_cluster_pairs = set()

    for step in range(1, threshold_steps + 1):     # Iterate through possible threshold values to find the one that maximizes F1_star
        t = step / threshold_steps
        cluster_pairs = cluster_products(distance_matrix, t)
        _, _, _, _, _, F1_star, _, _ = evaluate_performance(LSH_pairs, cluster_pairs, df)
        if F1_star > best_F1_star:
            best_F1_star = F1_star
            best_threshold = t
            best_cluster_pairs = cluster_pairs

    cluster_threshold_cache[cache_key] = (best_threshold, best_F1_star, best_cluster_pairs)
    return best_threshold, best_F1_star, best_cluster_pairs



#                   MAIN                             #


if __name__ == "__main__":
    logging.info("Loading data...")
    data = load_json_data(JSON_PATH)
    full_data = extract_data_from_json(data)
    full_data = clean_data_frame(full_data)
    full_data = full_data.reset_index(drop=True)

    results = []

    for boot_i in range(N_BOOTSTRAPS):     # Iterate through each bootstrap sample
        logging.info(f"Bootstrap {boot_i + 1}/{N_BOOTSTRAPS}")
        train_data, test_data = bootstrap_sample(full_data, boot_i)
        test_data = test_data.reset_index(drop=True)

        vocab = build_vocabulary(test_data)
        boolean_mat = create_binary_matrix(test_data, vocab)

        signatures = compute_minhash_signatures(boolean_mat, N_HASHES)

        for t in LSH_THRESHOLD_VALUES:
            start_time = time.time()

            candidate_pairs, b, r = locality_sensitive_hashing(signatures, t)

            if len(candidate_pairs) == 0:
                # No candidate pairs at all, skip clustering
                PQ, PC, F1, precision, recall, F1_star, frac_comparison, frac_cluster = 0, 0, 0, 0, 0, 0, 0, 0
            else:
                dist_matrix = build_dissimilarity_matrix(candidate_pairs, test_data)

                # Find optimal clustering threshold based on F1_star
                optimal_t, optimal_F1_star, optimal_cluster_pairs = optimal_threshold(dist_matrix, candidate_pairs,
                                                                                      test_data)

                # Evaluate performance with optimal threshold
                PQ, PC, F1, precision, recall, F1_star, frac_comparison, frac_cluster = evaluate_performance(
                    candidate_pairs, optimal_cluster_pairs, test_data
                )

            elapsed = time.time() - start_time

            results.append({
                'bootstrap': boot_i,
                'threshold': t,
                'b': b,
                'r': r,
                'PQ': PQ,
                'PC': PC,
                'F1': F1,
                'precision': precision,
                'recall': recall,
                'F1_star': F1_star,
                'fraction_comparison': frac_comparison,
                'fraction_cluster': frac_cluster,
                'time': elapsed
            })

    # Convert results to DataFrame
    res_df = pd.DataFrame(results)
    logging.info("Saving results to CSV...")
    res_df.to_csv("results_improved_code_with_caching_and_attr.csv", index=False)

    avg_res = res_df.groupby('threshold').mean().reset_index()
    #        Visualization        #
    plt.figure()
    plt.plot(avg_res['fraction_comparison'], avg_res['PC'], marker='o', linestyle='-', color='blue', label='Pair Completeness')
    plt.xlabel("Fraction of Comparisons")
    plt.ylabel("Pair Completeness (Recall at LSH stage)")
    plt.title("Pair Completeness vs Fraction of Comparisons")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(avg_res['fraction_comparison'], avg_res['PQ'], marker='o', linestyle='-', color='green', label='Pair Quality')
    plt.xlabel("Fraction of Comparisons")
    plt.ylabel("Pair Quality (Precision at LSH stage)")
    plt.title("Pair Quality vs Fraction of Comparisons")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(avg_res['fraction_comparison'], avg_res['F1'], marker='o', linestyle='-', color='red', label='F1 (LSH Blocking)')
    plt.xlabel("Fraction of Comparisons")
    plt.ylabel("F1 (LSH Blocking)")
    plt.title("F1 (LSH) vs Fraction of Comparisons")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(avg_res['fraction_comparison'], avg_res['F1_star'], marker='o', linestyle='-', color='purple', label='F1_star (Clustering)')
    plt.xlabel("Fraction of Comparisons")
    plt.ylabel("F1_star (Clustering)")
    plt.title("F1_star (Final Duplicate Detection) vs Fraction of Comparisons")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(avg_res['fraction_comparison'], avg_res['time'], marker='o', linestyle='-', color='orange', label='Time (seconds)')
    plt.xlabel("Fraction of Comparisons")
    plt.ylabel("Time (seconds)")
    plt.title("Time vs Fraction of Comparisons")
    plt.grid(True)
    plt.legend()
    plt.show()

# Convert the CSV results to an Excel file
csv_file = "/Users/henribotermans/Library/Mobile Documents/com~apple~CloudDocs/Rotterdam/Period 2/Computer Science/Papers/Final/results_improved_code_with_caching_and_attr.csv"  # Adjust path to your csv dataset
df = pd.read_csv(csv_file)
excel_file = "results_improved_code_with_caching_and_attr.xlsx"
df.to_excel(excel_file, index=False)
print(f"Results saved to {excel_file}")
