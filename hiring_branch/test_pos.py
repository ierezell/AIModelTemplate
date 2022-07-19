from hiring_branch.pos import get_part_of_speech


def test_simple():
    sentence = "I am a student."
    assert get_part_of_speech(sentence) == [
        ("I", "PRON"),
        ("am", "AUX"),
        ("a", "DET"),
        ("student", "NOUN"),
        (".", "PUNCT"),
    ]
    assert True
