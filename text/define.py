from text.symbols import symbols, common_symbols, en_symbols, zh_symbols

"""
0: en   1: zh   2: fr   3:de
4: ru   5: es   6: jp   8:ko
"""
def get_phoneme_set(path, encoding='utf-8'):
    phns = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            if line == '\n':
                continue
            phns.append('@' + line.strip())
    return phns


LANG_ID2SYMBOLS = {
    0: en_symbols,
    1: zh_symbols,
    2: common_symbols + get_phoneme_set("../MFA/lexicon/French/phoneset.txt"),
    3: common_symbols + get_phoneme_set("../MFA/lexicon/German/phoneset_css10-de.txt"),
    4: [],
    5: common_symbols + get_phoneme_set("../MFA/lexicon/Spanish/phoneset.txt"),
    6: common_symbols + get_phoneme_set("../MFA/lexicon/Japanese/phoneset.txt"),
    7: common_symbols + get_phoneme_set("../MFA/lexicon/Czech/phoneset.txt"),
    8: common_symbols + get_phoneme_set("../MFA/lexicon/Korean/phoneset.txt"),
}


LANG_ID2NAME = {
    0: "en",
    1: "zh",
    2: "fr",
    3: "de",
    4: "ru",
    5: "es",
    6: "jp",
    7: "cz",
    8: "ko"
}
