################
# Author: Kwok Keith
# Last Edit: 22 November 2025
################
import nltk
import json
from collections import defaultdict
from typing import Dict, List
from nltk.tokenize import RegexpTokenizer
import os


####################################
# FUNCTIONAL WORD MAP LOADERS      #
####################################

def load_category_word_map(path: str) -> Dict[str, List[str]]:
    """
    Load the category to functional words mapping from a JSON file.
    JSON format:
        {
            "category_1": ["word1", "word2", ...],
            "category_2": ["word3", "word4", ...]
        }
    """
    with open(path, "r") as f:
        category_word_map: Dict[str, List[str]] = json.load(f)
    return category_word_map


def build_word_category_map(category_word_map: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Convert mapping from category -> [words] to word -> category.
    """
    word_category_map: Dict[str, str] = {}
    for category, words in category_word_map.items():
        for word in words:
            word_category_map[word] = category
    return word_category_map


####################################
# TOKENISER                        #
####################################

def create_tokeniser() -> RegexpTokenizer:
    """
    Create a tokeniser that keeps A to Z, a to z, digits, and apostrophes.
    """
    # Corrected range: A-Za-z rather than A-za-z
    return nltk.tokenize.RegexpTokenizer(r"[A-Za-z0-9']+")


def tokenise(text: str, tokeniser: RegexpTokenizer) -> List[str]:
    """
    Lowercase and tokenise input text.
    """
    return tokeniser.tokenize(text.lower())


####################################
# CATEGORY COUNTING                #
####################################

def count_functional_categories(
    tokens: List[str],
    word_category_map: Dict[str, str]
) -> Dict[str, int]:
    """
    Given a list of tokens and a word -> category map,
    count how many tokens fall into each category.
    """
    category_count: Dict[str, int] = defaultdict(int)

    for token in tokens:
        category = word_category_map.get(token)
        if category is not None:
            category_count[category] += 1

    return dict(category_count)


####################################
# JSON DATA UTILITIES              #
####################################

def extract_conversation_json(json_path) -> tuple[list[str], list[str]]:
    """
    Load a conversation JSON file and extract the content of
    user and model messages into two ordered lists.

    Returns:
        user_chat:  list of message contents where agent == "user"
        model_chat: list of message contents where agent == "model"
    """
    # Load the JSON structure
    with open(json_path, "r", encoding="utf-8") as f:
        conversation = json.load(f)

    # Lists to hold ordered contents
    user_chat: list[str] = []
    model_chat: list[str] = []

    # Preserve the order of the messages array
    for message in conversation.get("messages", []):
        agent = message.get("agent")
        content = message.get("content", "")

        if agent == "user":
            user_chat.append(content)
        elif agent == "model":
            model_chat.append(content)

    return user_chat, model_chat

