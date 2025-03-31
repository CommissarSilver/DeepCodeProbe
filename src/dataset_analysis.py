import argparse
import os
import pickle
import warnings

import pandas as pd
import pickle5 as pickle
from tqdm import tqdm
import json
import glob

# Import datasets library for Hugging Face access
try:
    from datasets import load_dataset
except ImportError:
    print("Warning: `datasets` library not found. pip install datasets")
    load_dataset = None

warnings.filterwarnings("ignore")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    prog="Dataset Clone Distribution Analysis",
    description="Analyzes the distribution of clone/similarity labels in various datasets.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model whose dataset to analyze",
    choices=[
        "ast_nn",
        "funcgnn",
        "summarization_tf",  # Uses ast_nn java
        "code_sum_drl",
        "infercode",  # Uses ast_nn
        "recoder",  # Uses ast_nn java
        "type4py",
        "dear",  # Uses ast_nn java
    ],
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Base path to the datasets directory",
    default=os.path.join(os.getcwd(), "src"),
)
parser.add_argument(
    "--language",
    type=str,
    help="Language of the dataset. Required for models using ast_nn data (ast_nn, infercode, recoder, dear, summarization_tf)",
    choices=["java", "c"],
    default=None,  # Default to None, check requirement later
)

args = parser.parse_args()


# --- Helper Functions ---
def print_distribution(df: pd.DataFrame, label_col: str, dataset_name: str):
    """Calculates and prints the label distribution."""
    print(f"--- Analyzing: {dataset_name} ---")
    if label_col not in df.columns:
        print(f"Error: Label column '{label_col}' not found in the DataFrame.")
        return
    if df.empty:
        print("Error: DataFrame is empty.")
        return

    counts = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100

    print("Label Counts:")
    print(counts)
    print("\nLabel Percentages:")
    print(percentages)
    print("-" * (len(dataset_name) + 16))
    print("\\n")


# --- Main Analysis Logic ---

print(f"Starting analysis for model: {args.model}")

if args.model in ["ast_nn", "infercode", "recoder", "dear", "summarization_tf"]:
    if args.language is None:
        raise ValueError(
            f"Language argument (--language) is required for model '{args.model}'"
        )
    if args.model == "summarization_tf" and args.language != "java":
        raise ValueError(f"Summarization TF only supports Java dataset analysis.")
    if args.model == "recoder" and args.language != "java":
        raise ValueError(f"Recoder only supports Java dataset analysis.")
    if args.model == "dear" and args.language != "java":
        raise ValueError(f"DEAR only supports Java dataset analysis.")

    print(f"Using language: {args.language}")
    model_dataset_path = os.path.join(
        args.dataset_path, "ast_nn", "dataset", args.language
    )
    clone_ids_path = os.path.join(model_dataset_path, "clone_ids.pkl")
    programs_path_pkl = os.path.join(model_dataset_path, "programs.pkl")
    programs_path_tsv = os.path.join(model_dataset_path, "programs.tsv")

    if not os.path.exists(clone_ids_path):
        raise FileNotFoundError(f"Clone IDs file not found: {clone_ids_path}")

    print(f"Loading clone IDs from: {clone_ids_path}")
    clone_ids = pickle.load(open(clone_ids_path, "rb"))

    programs = None
    if args.language == "c":
        if not os.path.exists(programs_path_pkl):
            raise FileNotFoundError(f"Programs file not found: {programs_path_pkl}")
        print(f"Loading C programs from: {programs_path_pkl}")
        programs = pickle.load(open(programs_path_pkl, "rb"))
        programs.columns = [
            "id",
            "code",
            "label",
        ]  # C dataset has label in programs.pkl
        programs.drop(
            columns=["label"], inplace=True
        )  # Drop it as we get label from clone_ids
    elif args.language == "java":
        if not os.path.exists(programs_path_tsv):
            raise FileNotFoundError(f"Programs file not found: {programs_path_tsv}")
        print(f"Loading Java programs from: {programs_path_tsv}")
        programs = pd.read_csv(
            programs_path_tsv, delimiter="\\t", engine="python", on_bad_lines="skip"
        )
        programs.columns = [
            "id",
            "code",
        ]  # Java dataset does not have label in programs.tsv

    if programs is None:
        raise ValueError("Failed to load programs data.")

    print("Merging clone IDs and programs data...")
    clone_ids["id1"] = clone_ids["id1"].astype(int)
    clone_ids["id2"] = clone_ids["id2"].astype(int)

    # Ensure program IDs are integers for merging
    programs["id"] = pd.to_numeric(programs["id"], errors="coerce")
    programs.dropna(
        subset=["id"], inplace=True
    )  # Remove rows where ID couldn't be converted
    programs["id"] = programs["id"].astype(int)

    # Perform the merges
    merged_data = pd.merge(
        clone_ids, programs, how="left", left_on="id1", right_on="id"
    )
    merged_data = pd.merge(
        merged_data,
        programs,
        how="left",
        left_on="id2",
        right_on="id",
        suffixes=("_x", "_y"),
    )

    # Clean up
    merged_data.drop(
        ["id_x", "id_y"], axis=1, inplace=True, errors="ignore"
    )  # Use errors='ignore' in case columns are already dropped
    merged_data.dropna(
        subset=["code_x", "code_y", "label"], inplace=True
    )  # Drop rows with missing essential data
    merged_data.reset_index(drop=True, inplace=True)

    print_distribution(
        merged_data,
        "label",
        f"AST-NN ({args.language.upper()}) derived dataset for {args.model}",
    )

elif args.model == "funcgnn":
    print("Analyzing FuncGNN dataset...")
    model_dataset_path = os.path.join(args.dataset_path, "funcgnn", "dataset")
    train_files_path = os.path.join(model_dataset_path, "train")
    test_files_path = os.path.join(
        model_dataset_path, "test"
    )  # Although test might not have explicit labels

    if not os.path.isdir(train_files_path):
        raise FileNotFoundError(
            f"FuncGNN train directory not found: {train_files_path}"
        )

    train_files = glob.glob(os.path.join(train_files_path, "*.json"))
    if not train_files:
        print("Warning: No JSON files found in FuncGNN train directory.")
        # Decide how to proceed: maybe exit or just report zero counts

    similars_map = {}  # Map prefix to list of full file IDs
    dissimilar_pairs = []  # List of [id1, id2] pairs

    print("Processing FuncGNN train files to identify similar/dissimilar pairs...")
    for file_path in tqdm(train_files):
        base_name = os.path.basename(file_path)
        file_name_no_ext = base_name.replace(".json", "")
        try:
            id1, id2 = file_name_no_ext.split("::::")
            dissimilar_pairs.append(
                [id1, id2]
            )  # Assume files in train represent dissimilar pairs directly as per validate_probe logic

            # Logic for similars based on common prefix (from validate_probe)
            prefix1 = id1.split("_")[0]
            if prefix1 not in similars_map:
                similars_map[prefix1] = set()
            similars_map[prefix1].add(id1)

            prefix2 = id2.split("_")[0]
            if prefix2 not in similars_map:
                similars_map[prefix2] = set()
            similars_map[prefix2].add(id2)

        except ValueError:
            print(f"Warning: Could not split filename into pair: {base_name}")
            continue

    # Count similar pairs (pairs within the same prefix group)
    similar_count = 0
    for prefix, ids_set in similars_map.items():
        ids_list = list(ids_set)
        # Number of pairs in a group of size n is n * (n - 1) / 2
        if len(ids_list) > 1:
            similar_count += len(ids_list) * (len(ids_list) - 1) // 2

    dissimilar_count = len(dissimilar_pairs)
    total_pairs = similar_count + dissimilar_count

    print(f"--- Analyzing: FuncGNN ---")
    print("Label Counts (Inferred):")
    print(f"Similar Pairs: {similar_count}")
    print(f"Dissimilar Pairs: {dissimilar_count}")
    print(f"Total Pairs: {total_pairs}")

    if total_pairs > 0:
        print("\nLabel Percentages (Inferred):")
        print(f"Similar Pairs: {similar_count / total_pairs * 100:.2f}%")
        print(f"Dissimilar Pairs: {dissimilar_count / total_pairs * 100:.2f}%")
    else:
        print("\nNo pairs found to calculate percentages.")
    print("-" * 28)  # Length of "--- Analyzing: FuncGNN ---"
    print("\\n")


elif args.model == "code_sum_drl":
    print("Analyzing CodeSearchNet (CodeSumDRL / PoolC) dataset...")
    # Define potential local paths
    csv_path = os.path.join(args.dataset_path, "code_sum_drl", "dataset", "clones.csv")
    alt_csv_path = os.path.join(
        args.dataset_path, "code_sum_drl", "dataset_clones", "clones.csv"
    )

    df = None  # Initialize DataFrame

    # --- Attempt to load from local CSV first ---
    load_path = None
    if os.path.exists(csv_path):
        load_path = csv_path
    elif os.path.exists(alt_csv_path):
        load_path = alt_csv_path

    if load_path:
        print(f"Loading data from local CSV: {load_path}")
        try:
            df = pd.read_csv(load_path)
            # Check if required columns exist after loading CSV
            required_cols_csv = ["code_x", "code_y", "label"]  # Expected names in CSV
            if not all(col in df.columns for col in required_cols_csv):
                print(
                    f"Warning: CSV {load_path} might not have expected columns {required_cols_csv}. Found: {df.columns.tolist()}"
                )
                # Attempt to handle potential column name variations if necessary
                # For now, we assume the CSV has the right columns or processing below handles it.

        except Exception as e:
            print(f"Error reading local CSV {load_path}: {e}")
            df = None  # Ensure df is None if read fails

    # --- Fallback to Hugging Face if local load failed or file not found ---
    if df is None:
        print(f"Local CSV not found or failed to load.")
        if load_dataset is not None:
            print(
                "Attempting to load from Hugging Face: PoolC/1-fold-clone-detection-600k-5fold"
            )
            try:
                # Load the dataset (only train split is relevant here)
                dataset_hf = load_dataset(
                    "PoolC/1-fold-clone-detection-600k-5fold", split="train"
                )
                print("Converting Hugging Face dataset to DataFrame...")
                df = dataset_hf.to_pandas()
                print("Processing DataFrame columns...")
                # Drop unnecessary columns mentioned in validate_probe.py comments
                cols_to_drop = [
                    "pair_id",
                    "question_pair_id",
                    "code1_group",
                    "code2_group",
                ]
                df.drop(
                    columns=[col for col in cols_to_drop if col in df.columns],
                    inplace=True,
                )

                # Rename columns to match the expected format ["code_x", "code_y", "label"]
                # Based on validate_probe.py comments, the original HF dataset likely has "code1", "code2", "label"
                rename_map = {
                    "code1": "code_x",
                    "code2": "code_y",
                    "similar": "label",  # Add mapping for the label column
                }
                actual_rename_map = {
                    k: v for k, v in rename_map.items() if k in df.columns
                }
                df.rename(columns=actual_rename_map, inplace=True)
                print("DataFrame columns processed.")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load or process dataset from Hugging Face: {e}. Please check network connection and `datasets` installation."
                )
        else:
            raise RuntimeError(
                "Local CSV not found and `datasets` library is not available to download from Hugging Face."
            )

    # --- Final Check and Distribution Analysis ---
    if df is None:
        # This should theoretically not be reached due to prior error handling
        raise ValueError("Could not load dataset from local CSV or Hugging Face.")

    # Validate required columns *after* loading and processing from either source
    required_cols = ["code_x", "code_y", "label"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Dataset (from CSV or HF) is missing required columns after processing. Found: {df.columns.tolist()}. Expected: {required_cols}"
        )

    print_distribution(df, "label", "CodeSumDRL (PoolC/CodeSearchNet Clones)")


elif args.model == "type4py":
    print("Analyzing Type4Py (Python Clones) dataset...")
    # Path based on validate_probe.py
    csv_path = os.path.join(
        args.dataset_path, "code_sum_drl", "python_clones.csv"
    )  # Uses the file from code_sum_drl dir

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Type4Py python_clones.csv not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise IOError(f"Failed to load Type4Py CSV {csv_path}: {e}")

    # Expected columns based on validate_probe: ['code1', 'code2', 'similar'] (1=similar, 0=dissimilar)
    required_cols = ["code1", "code2", "similar"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"CSV file {csv_path} is missing required columns. Found: {df.columns.tolist()}. Expected: {required_cols}"
        )

    print_distribution(df, "similar", "Type4Py (Python Clones)")

else:
    print(f"Analysis logic for model '{args.model}' is not yet implemented.")

print("Dataset analysis finished.")
