
from __future__ import annotations

import time
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
from BO import main_bo



# -----------------------
# Helpers
# -----------------------
def update_results_files(
    *,
    labels_path: Path,
    df_path: Path,
    query_row: int,
    fom_value: float,
    nl_threshold: float = 0.02,
    label_col: str = "value",
) -> None:
    """Write FOM into labels.csv + processed.csv, and refresh linearity_flag."""
    labels = pd.read_csv(labels_path)
    df = pd.read_csv(df_path)

    labels.loc[query_row, label_col] = fom_value
    df.loc[query_row, label_col] = fom_value

    df["NL_score"] = (df[label_col] - df["linear_value"]) / df["linear_value"]
    low_nl = df[df["NL_score"] < nl_threshold]

    def _to_tuple(x):
        if isinstance(x, str):
            # safe parsing without eval
            import ast
            return tuple(ast.literal_eval(x))
        return tuple(x)

    cmp_ids_to_flag = set(tuple(sorted(_to_tuple(ids))) for ids in low_nl["cmp_ids"])
    df["linearity_flag"] = df["cmp_ids"].apply(lambda x: tuple(sorted(_to_tuple(x))) in cmp_ids_to_flag)

    labels.to_csv(labels_path, index=False)
    df.to_csv(df_path, index=False)


def save_top_features(top_features, iteration_i: int, base_dir: Path = Path("FAMBO/results/features")) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = base_dir / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now().strftime("%H%M")
    out_path = out_dir / f"{iteration_i}_{run_tag}.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(top_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path


# -----------------------
# Main wrapper
# -----------------------
def run_sdl_bo(args: BOArgs, chemicals: list[str]) -> None:

    assert args.nb_iterations > args.n_data_gathering, (
        f"nb_iterations ({args.nb_iterations}) must be > n_data_gathering ({args.n_data_gathering})."
    )

    for i in range(args.nb_iterations):
        print("\n" + "+" * 24)
        print(f"Iteration {i+1}/{args.nb_iterations}")

        is_gathering = i < args.n_data_gathering
        which_acq = "max sigma" if is_gathering else "EI"

        if is_gathering:
            print("Mode: data gathering (max sigma)")
            if i == 0:
                labels = pd.read_csv(args.labels_path)
                n_data = labels[args.label].notna().sum()
                max_allowed = 26 + args.n_data_gathering
                assert n_data <= max_allowed, (
                    f"'{args.label}' has {n_data} entries, allowed max is {max_allowed}."
                )
        else:
            print("Mode: BO (EI)")

        # Call BO (same return signature)
        # IMPORTANT: main_bo reads args.which_acquisition, so set it just for this call
        old_acq = args.which_acquisition
        args.which_acquisition = which_acq
        comp1, comp2, comp3, x1, x2, x3, queries_ids, top_features = main_bo(args)
        args.which_acquisition = old_acq  # restore

        # Inventory + logging (your functions)
        # check_remaining(INVENTORY_XLSX, comp1, comp2, comp3)
        # time.sleep(3)
        # log_usage(INVENTORY_XLSX, comp1, comp2, comp3, x1, x2, x3)

        # Lab execution (your function)
        # fom = datagather(comp1, comp2, comp3, x1, x2, x3, chemicals, queries_ids)
        fom = 0.0  # placeholder for actual FOM from lab

        # Persist results + recompute linearity flags
        query_row = int(queries_ids[0])
        update_results_files(
            labels_path=args.labels_path,
            df_path=args.df_path,
            query_row=query_row,
            fom_value=fom,
            nl_threshold=0.02,
            label_col=args.label,
        )

        # Save selected feature set for this iteration
        out_path = save_top_features(top_features, iteration_i=i+1)
        print(f"Saved top_features to: {out_path}")


