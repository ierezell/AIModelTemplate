from aimodel.sentence_splitter import split_sentence


def test_complex_sentence():
    sentence = "Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e.   he paid a lot   for it.    Did he mind! !!  !! Maybe not ?Adam Jones Jr. thinks he didn't... In any case, this isn't true!... Well, with a probability of .9! It isn't."
    splitted_sentences = split_sentence(sentence)
    assert splitted_sentences == [
        "Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e.   he paid a lot   for it.",
        "Did he mind! !!  !!",
        "Maybe not ?",
        "Adam Jones Jr. thinks he didn't...",
        "In any case, this isn't true!...",
        "Well, with a probability of .9!",
        "It isn't.",
    ]


def test_simple_sentence():
    sentence = "Smith bought cheapsite for one million dollars. He paid a lot for it."
    splitted_sentences = split_sentence(sentence)
    assert splitted_sentences == [
        "Smith bought cheapsite for one million dollars.",
        "He paid a lot for it.",
    ]
