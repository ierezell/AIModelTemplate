import re

STOPS_REGEX = re.compile(
    #   no Mr./Jr./      no i.e.   or  1+ of .?!;  a space and a letter
    r"(?<![A-Z][a-z]\.)(?<!\w\.\w\.)(?<=[.?!;])\s+(?=\w)",
)
PUNCTUATION_REGEX = re.compile(r"[?!]\w")  # Split sentences like "Hello!Adam replied."


def split_sentence(string: str) -> list[str]:
    sentences: list[str] = []
    cursor_index: int = 0

    for stop in STOPS_REGEX.finditer(string):
        sent = string[cursor_index : stop.start()]

        sub_cursor_index = 0
        sub_sentences: list[str] = []
        for sub_stop in PUNCTUATION_REGEX.finditer(sent):
            sub_sent = sent[sub_cursor_index : sub_stop.start() + 1]
            sub_sentences.append(sub_sent)
            sub_cursor_index = sub_stop.end() - 1

        sub_sentences.append(sent[sub_cursor_index:])
        sentences.extend(sub_sentences)

        cursor_index = stop.end()
    sentences.append(string[cursor_index:])

    return sentences
