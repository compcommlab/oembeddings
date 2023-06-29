import re

LINK_REGEX = re.compile('\b(?:https?:\/\/|www\.)(?:[\w-]+\.)*[\w-]+(?:\.(?:at|com|de|co))\b')
EMAIL_REGEX = re.compile(r'\S+@\S+')
QUOTATION_MARKS = re.compile(r"([“”‘’«»„“”‹›❝❞〝〞〟＂＇«»‹›〈〉《》「」『』【】〔〕〖〗〘〙〚〛‘’‚‘’‛“”„“”‟‘’“”‘’„“”‹›«»«»“”»«»“”»«»“”‹›‘’‘’‘’‚‘’‚‘’‛‘’‛‘’“”„“”„“”‟„“”‟‘’‹›‹›])")
PUNCTUATION = re.compile(r'([\!\"#$%&\'()*+,-–\.…/:;<=>\?@\[\\\]^_`\{\|\}\~])')
NUMBERS = re.compile(r'\d+')
WHITESPACE = re.compile(r'\s{2,}')
LINE_BREAKS = re.compile(r'\n')


if __name__ == "__main__":
    print('testing regex')

    t1 = "Amazon prüft „weitere Konsequenzen“."
    assert PUNCTUATION.sub(" ", t1) == "Amazon prüft  weitere Konsequenzen  ", PUNCTUATION.sub(" ", t1)