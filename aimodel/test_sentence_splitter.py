from aimodel.sentence_splitter import split_sentence


def test_complex_sentence() -> None:
    sentence = (
        "Mr. Smith bought cheap_site.com for 1.5 million dollars, i.e.   "
        "he paid a lot   for it.    Did he mind! !!  !! Maybe not ?"
        "Adam Jones Jr. thinks he didn't... In any case, this isn't true!... "
        "Well, with a probability of .9! It isn't."
    )
    splitted_sentences = split_sentence(sentence)
    assert splitted_sentences == [
        "Mr. Smith bought cheap_site.com for 1.5 million dollars, i.e.   he paid a lot   for it.",  # noqa: E501
        "Did he mind! !!  !!",
        "Maybe not ?",
        "Adam Jones Jr. thinks he didn't...",
        "In any case, this isn't true!...",
        "Well, with a probability of .9!",
        "It isn't.",
    ]


def test_simple_sentence() -> None:
    sentence = "Smith bought cheap site for one million dollars. He paid a lot for it."
    splitted_sentences = split_sentence(sentence)
    assert splitted_sentences == [
        "Smith bought cheap site for one million dollars.",
        "He paid a lot for it.",
    ]
