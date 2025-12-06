################
# Author: Kwok Keith
# Last Edit: 06 December 2025
################
from lsm_api import lsm
import os
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LSM scores for a conversation JSON file"
    )

    parser.add_argument(
        "--fw",
        type=str,
        default=os.path.join(os.curdir, "functional_word.json"),
        help=(
            "Path to functional_word.json. "
            "Defaults to ./functional_word.json"
        ),
    )

    parser.add_argument(
        "--i",
        type=str,
        required=True,
        help=(
            "Path to the conversation JSON file. "
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Paths to json datasets
    path_to_functional_word_json = args.fw
    path_to_conversation_json = args.i

    # Call lsm API
    output = lsm(
        path_to_functional_word_json,
        path_to_conversation_json,  
        smoothing_window=5,
        min_chat_token=10
    )

    # Get output values
    static_lsm = output["static_lsm"]
    lsm_per_turn = output["lsm_per_turn"]
    lsm_per_turn_cat = output["lsm_per_turn_cat"]

    print(f"Static LSM: {static_lsm:.4f}")
    for t, score in enumerate(lsm_per_turn):
        print(f"Turn {t}: LSM_t = {score:.4f}")


if __name__ == "__main__":
    main()
