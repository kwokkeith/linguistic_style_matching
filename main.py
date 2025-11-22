################
# Author: Kwok Keith
# Last Edit: 22 November 2025
################
from library import *
from lsm_library import *
import os

def main() -> None:
    # Path to functional_word.json relative to current working directory
    path_to_functional_word_json = os.path.join(os.curdir, "functional_word.json")
    path_to_conversation_json = os.path.join(os.curdir, "gemini_conversation_001.json")

    # Load maps
    category_word_map = load_category_word_map(path_to_functional_word_json)
    word_category_map = build_word_category_map(category_word_map)
    categories = sorted(category_word_map.keys()) 

    # Create tokeniser
    tokeniser = create_tokeniser()

    # Extract out the user convo and model content from data json
    user_chat, model_chat = extract_conversation_json(path_to_conversation_json)

    static_lsm, lsm_per_turn, lsm_per_turn_cat = compute_lsm(
        user_chat=user_chat,
        model_chat=model_chat,
        tokeniser=tokeniser,
        word_category_map=word_category_map,
        categories=categories,
        window=3,        # last X chats (For smoothing)
        min_tokens=10,   # Ignore short turns (lack of info)
    )

    print(f"Static LSM: {static_lsm:.4f}")
    for t, score in enumerate(lsm_per_turn):
        print(f"Turn {t}: LSM_t = {score:.4f}")


if __name__ == "__main__":
    main()
