import spacy
from spacy.matcher import Matcher

from hiring_branch.pos import PosType

nlp = spacy.load("en_core_web_lg")


def is_passive_sentence_rule_base(tokens: list[tuple[str, PosType]]):
    for tok_idx in range(len(tokens) - 1):
        if tokens[tok_idx][1] == "VERB" and tokens[tok_idx + 1][1] == "PROPN":
            return True
    return False


def is_passive_sentence(sentence: str):

    matcher = Matcher(nlp.vocab)
    passive_rule = [
        {"DEP": {"IN": ["nsubjpass", "xcomp", "aux"]}, "OP": "*"},
        {"DEP": "auxpass"},
        {"DEP": "nsubj", "OP": "*"},
        {"TAG": "VBN"},
    ]

    matcher.add("PassiveRule1", [passive_rule])

    if matcher(nlp(sentence)):
        return True
    return False
