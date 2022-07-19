from hiring_branch.passive import is_passive_sentence


def test_passive():
    sentence = "At dinner, six shrimp were eaten by Harry."
    assert is_passive_sentence(sentence)


def test_non_passive():
    sentence = "Harry ate six shrimp at dinner."
    assert not is_passive_sentence(sentence)
