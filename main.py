################
# Author: Kwok Keith
# Last Edit: 06 December 2025
################
from library import *
from lsm_library import *
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

    # Load maps
    category_word_map = load_category_word_map(path_to_functional_word_json)
    word_category_map = build_word_category_map(category_word_map)
    categories = sorted(category_word_map.keys())

    # Create tokeniser
    tokeniser = create_tokeniser()

    # Extract out the user convo and model content from data json
    user_chat, model_chat = extract_conversation_json(
        path_to_conversation_json)

    static_lsm, lsm_per_turn, lsm_per_turn_cat = compute_lsm(
        user_chat=user_chat,
        model_chat=model_chat,
        tokeniser=tokeniser,
        word_category_map=word_category_map,
        categories=categories,
        window=5,        # last X chats (For smoothing)
        min_tokens=10,   # Ignore short turns (lack of info)
    )

    print(f"Static LSM: {static_lsm:.4f}")
    for t, score in enumerate(lsm_per_turn):
        print(f"Turn {t}: LSM_t = {score:.4f}")


if __name__ == "__main__":
    main()
