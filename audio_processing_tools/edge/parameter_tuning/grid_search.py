import os, glob, json, hashlib, time
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Callable, Dict
import pandas as pd
from tqdm import tqdm


def load_processed_param_ids(pattern):
    """Loads existing results from files matching the given pattern."""
    ids = []
    for filename in glob.glob(pattern):
        with open(filename, "r") as file:
            result = json.load(file)
            # Convert params_key to a string to ensure it is hashable
            params_key = str(tuple(result["parameters"].items()))
            ids.append(params_key)
    return ids


def replace_callables(obj):
    """Recursively replace callable objects in a dictionary or list with their names."""
    if isinstance(obj, dict):
        return {k: replace_callables(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_callables(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(replace_callables(v) for v in obj)
    elif callable(obj):
        return obj.__name__  # Replace the callable with its name
    else:
        return obj
def save_result_to_disk(result, filename):
    """Saves a single result to a JSON file."""
    # remove non serializable callables
    result = replace_callables(result)

    with open(filename, "w") as f:
        json.dump(result, f, indent=4)


def params_to_filename(params_key, alg_identifier):
    # Generate a hash of the parameters
    params_hash = hashlib.sha256(params_key.encode()).hexdigest()
    # Use the hash in the filename
    filename = f"{alg_identifier}_{params_hash}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    return filename


def grid_search(
    audio_df: pd.DataFrame,
    custom_alg: Callable[[pd.DataFrame, Any], tuple],
    param_grid: Dict[str, list],
    test_name: str,
    results_dir: str,
) -> None:
    """
    Executes a grid search over a set of parameters for a custom algorithm, using audio data.

    Parameters:
    - audio_df (pd.DataFrame): The DataFrame containing audio data for the algorithm.
    - custom_alg (Callable[[pd.DataFrame, Any], tuple]): The custom algorithm to be optimized. It should accept the audio DataFrame and parameters, returning a tuple with overall accuracy and classification results.
    - param_grid (Dict[str, list]): A dictionary where keys are parameter names and values are lists of parameter values to be tested.
    - experiment_identifier (str): A unique identifier for the algorithm or other compoents of the experiment, used in saving results.
    - results_dir (str, optional): The directory where result files will be saved.

    The function iterates over every combination of parameters in `param_grid`, evaluates the algorithm's performance, and saves the results to disk.
    It skips the evaluation of parameter combinations that have already been processed, to avoid redundant computations.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load existing results to ensure there are not redundant computations
    existing_results = load_processed_param_ids(
        os.path.join(results_dir, f"{test_name}_*.json")
    )

    # Calculate the total number of combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    for combination in tqdm(product(*param_grid.values()), total=total_combinations):
        params = dict(zip(param_grid.keys(), combination))
        # Convert params_key to a string to ensure it is hashable
        params_key = str(tuple(params.items()))

        # Check if these parameters have been processed
        if params_key in existing_results:
            print(f"Skipping already processed combination: {params}")
            continue

        # Run the algorithm with the current set of parameters
        (
            overall_accuracy,
            true_positive_uids,
            true_negative_uids,
            false_positive_uids,
            false_negative_uids,
        ) = custom_alg(audio_df, **params)

        result = {
            "test_name": test_name,
            "parameters": params,
            "overall_accuracy": overall_accuracy,
            "tp_classifications": true_positive_uids,
            "tn_classifcations": true_negative_uids,
            "fp_classifications": false_positive_uids,
            "fn_classifications": false_negative_uids,
        }

        # Save the result to disk
        filename = params_to_filename(params_key, test_name)
        save_path = os.path.join(results_dir, filename)
        save_result_to_disk(result, save_path)
        print(f"Processed and saved: {params}")


def execute_algorithm(
    params_key: str,
    audio_df: pd.DataFrame,
    params: Dict[str, Any],
    experiment_identifier: str,
    results_dir: str,
    custom_alg: Callable,
) -> None:
    """
    Executes the custom algorithm with a set of parameters and saves the results.

    This function is intended to be run in a parallel execution context.
    """
    # Run the algorithm with the current set of parameters
    result_tuple = custom_alg(audio_df, **params)

    result = {
        "experiment": experiment_identifier,
        "parameters": params,
        "overall_accuracy": result_tuple[0],
        "tp_classifications": result_tuple[1],
        "tn_classifications": result_tuple[2],
        "fp_classifications": result_tuple[3],
        "fn_classifications": result_tuple[4],
    }

    # Save the result to disk
    filename = params_to_filename(params_key, experiment_identifier)
    save_path = os.path.join(results_dir, filename)
    save_result_to_disk(result, save_path)
    print(f"Processed and saved: {params}")


def grid_search_parallel(
    audio_df: pd.DataFrame,
    custom_alg: Callable[[pd.DataFrame, Any], tuple],
    param_grid: Dict[str, list],
    experiment_identifier: str,
    results_dir: str = "./parameter_search_results/",
    max_workers: int = None
) -> None:
    """
        Performs a parallel grid search to apply a custom algorithm with various parameter combinations on an audio DataFrame.

        Args:
            audio_df (pd.DataFrame): A DataFrame containing the audio data over which the algorithm will be applied.
            custom_alg (Callable[[pd.DataFrame, Any], tuple]): The custom algorithm function that accepts a DataFrame and parameter set.
            param_grid (Dict[str, list]): A dictionary where keys are the parameter names and values are lists of parameter settings to test.
            experiment_identifier (str): A unique identifier for the experiment used which will be reflected in the results filenames.
            results_dir (str, optional): The directory path where the results will be saved. Defaults to "./parameter_search_results/".
            max_workers (int, optional): The maximum number of parallel workers to use. If None, the number of threads is set to the number of processors.

        Returns:
            None: The function doesn't return any value. Results are written directly to the filesystem in the specified results directory.

        This function automates the grid search process by parallelizing the execution of the custom algorithm over all combinations
        of the parameters given in `param_grid` and saves the results to `results_dir`. A progress bar shows the overall progress
        of the computation. If results already exist for some parameter combinations, those are skipped to prevent redundant
        computations.

        Exceptions raised within the threads (from `custom_alg` execution) are caught and reported in the progress display.
        """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    existing_results = load_processed_param_ids(
        os.path.join(results_dir, f"{experiment_identifier}_*.json")
    )

    combinations = list(product(*param_grid.values()))
    print("Starting Parallel Jobs")
    print("Wait for first task to Complete for progress indicator. Will take ~ 1 min/1000 test vectors (after files are cached)")
    start_time = time.time()

    tasks = []
    # Filter combinations to submit only those that haven't been processed
    for combination in combinations:
        params = dict(zip(param_grid.keys(), combination))
        params_key_for_file_check = str(replace_callables(tuple(params.items())))
        params_key = str(tuple(params.items()))

        if params_key_for_file_check in existing_results:
            print(f"Already Processed {params}, skipping")
            continue  # Skip already processed combinations
        tasks.append((params_key, params))

    # Use the executor to submit all tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(tasks)) as progress_bar:
        future_to_params = {
            executor.submit(execute_algorithm, params_key, audio_df, params, experiment_identifier, results_dir, custom_alg): params
            for params_key, params in tasks
        }

        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                # Handle results or exceptions here
                future.result()
                progress_bar.update(1)
            except Exception as e:
                progress_bar.write(f"Error processing parameter combination {params}: {e}")
                progress_bar.update(1)
                raise

    end_time = time.time()
    print(f"Grid search completed in {end_time - start_time:.2f} seconds.")
