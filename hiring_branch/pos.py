from typing import Literal, cast

import spacy

PosType = Literal[
    "AUX",
    "VERB",
    "PROPN",
    "NOUN",
    "ADP",
    "SYM",
    "NUM",
]


nlp = spacy.load("en_core_web_lg")


def get_part_of_speech(sentence: str) -> list[tuple[str, PosType]]:
    return [(token.text, cast(PosType, token.pos_)) for token in nlp(sentence)]
