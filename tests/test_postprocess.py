from voxtray.postprocess import normalize_transcript


def test_normalize_transcript_basic_cleanup():
    text = "  Hola   mundo  ,  esto es una prueba  !\n"
    assert normalize_transcript(text) == "Hola mundo, esto es una prueba!"
