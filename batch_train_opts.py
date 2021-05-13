import argparse


def options():
    """Parse options for the batch-mode FCRN-A training script."""
    p = argparse.ArgumentParser(
        description="Run multiple FCRN-A trainings" + "in batch mode (i.e., in serial)"
    )
    p.add_argument(
        "trainParams",
        help="Options to pass to the training script"
        + " (note: enclose them in quotations)",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--n_repeats",
        help="Number of times to repeat the training. If --existing_nets is set,"
        " then this input is ignored; the trainings will run as long as needed for"
        " every existing net to be retrained.",
        type=int,
    )
    group.add_argument(
        "--existing_nets", help="Path to folder containing existing nets to retrain."
    )
    return p.parse_args()