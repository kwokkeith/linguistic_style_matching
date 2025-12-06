################
# Author: Kwok Keith
# Last Edit: 06 December 2025
################
from library import *
from lsm_library import *
import os

def lsm(path_to_functional_word_json: str, 
        path_to_conversation_json: str,
        smoothing_window: int = 5,
        min_chat_token: int = 10
        ) -> dict[str]:
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
        window=smoothing_window,      # last X chats (For smoothing)
        min_tokens=min_chat_token,    # Ignore short turns (lack of info)
    )

    # Generate output 
    output = {
        "static_lsm": static_lsm,
        "lsm_per_turn": lsm_per_turn,
        "lsm_per_turn_cat": lsm_per_turn_cat
    }
    
    return output
