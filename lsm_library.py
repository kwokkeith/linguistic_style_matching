################
# Author: Kwok Keith
# Last Edit: 22 November 2025
################
from typing import Optional, Tuple
from library import *

####################################
# LSM COMPUTATION                  #
####################################

def compute_proportion_vector(
    text: str,
    tokeniser: RegexpTokenizer,
    word_category_map: Dict[str, str],
    categories: List[str],
    min_tokens: int = 1,
) -> Optional[List[float]]:
    """
    Compute f_t in R^K for one utterance:
        f_{t,i} = count(tokens in category i) / total tokens in utterance
    """
    tokens = tokenise(text, tokeniser)
    if len(tokens) < min_tokens:
        return None
    
    total_tokens = len(tokens)
    if total_tokens == 0:
        return [0.0 for _ in categories]

    counts: Dict[str, int] = defaultdict(int)
    for tok in tokens:
        cat = word_category_map.get(tok)
        if cat is not None:
            counts[cat] += 1

    return [counts[cat] / total_tokens for cat in categories]


def build_proportion_series(
    chat: List[str],
    tokeniser: RegexpTokenizer,
    word_category_map: Dict[str, str],
    categories: List[str],
    min_tokens: int = 1,
) -> List[List[float]]:
    """
    For a sequence of utterances, build [f_0, f_1, ..., f_{T-1}]
    where each f_t is a K dimensional list of proportions.
    """
    return [
        compute_proportion_vector(utt, tokeniser, word_category_map, categories, min_tokens)
        for utt in chat
    ]


def moving_average_vectors(
    vectors: List[Optional[List[float]]],
    window: int,
) -> List[List[float]]:
    """
    Sliding window mean over time for each dimension.

    For each t, look back over the last `window` turns for that speaker
    (indices t, t-1, ..., t-window+1), but only average the entries that
    are not None.

    If there are no valid entries in the window, return a zero vector.
    """
    if not vectors:
        return []

    # Find first non None to get K (Can have None for filtered out chats)
    first_valid = next((v for v in vectors if v is not None), None)
    if first_valid is None:
        # No valid vectors at all
        return []

    K = len(first_valid)
    smoothed: List[List[float]] = []

    for t in range(len(vectors)):
        start = max(0, t - window + 1)
        window_vecs = [v for v in vectors[start : t + 1] if v is not None]

        if not window_vecs:
            avg_vec = [0.0 for _ in range(K)]
        else:
            w_len = len(window_vecs)
            avg_vec = [
                sum(vec[i] for vec in window_vecs) / w_len
                for i in range(K)
            ]

        smoothed.append(avg_vec)

    return smoothed


def compute_lsm(
    user_chat: List[str],
    model_chat: List[str],
    tokeniser: RegexpTokenizer,
    word_category_map: Dict[str, str],
    categories: List[str],
    window: int = 3,
    epsilon: float = 1e-6,
    min_tokens: int = 10,
) -> Tuple[float, List[float], List[List[float]]]:
    """
    Compute LSM using:

        f_t,i from proportion vectors
        windowed averages over the last `window` valid chats
        ignoring turns with < min_tokens tokens
    """
    T = min(len(user_chat), len(model_chat))
    if T == 0:
        return 0.0, [], []

    # Proportion series with short utterances marked as None
    user_f = build_proportion_series(
        user_chat[:T], tokeniser, word_category_map, categories, min_tokens=min_tokens
    )
    model_f = build_proportion_series(
        model_chat[:T], tokeniser, word_category_map, categories, min_tokens=min_tokens
    )

    # Sliding window means over last X chats per speaker
    user_f_bar = moving_average_vectors(user_f, window)
    model_f_bar = moving_average_vectors(model_f, window)

    K = len(categories)
    lsm_per_turn: List[float] = []
    lsm_per_turn_cat: List[List[float]] = []

    for t in range(T):
        cat_scores: List[float] = []
        for i in range(K):
            a = user_f_bar[t][i]
            b = model_f_bar[t][i]
            num = abs(a - b)
            denom = a + b + epsilon
            lsm_t_i = 1.0 - num / denom
            cat_scores.append(lsm_t_i)

        turn_score = sum(cat_scores) / K
        lsm_per_turn.append(turn_score)
        lsm_per_turn_cat.append(cat_scores)

    static_lsm = sum(lsm_per_turn) / T
    return static_lsm, lsm_per_turn, lsm_per_turn_cat