import re

# regular expressions to clean raw sentences.
# We follow the procedures that the Facebook engineers used for fasttext:
# References:
# https://github.com/facebookresearch/fastText/blob/main/tests/fetch_test_data.sh
# https://github.com/facebookresearch/fastText/blob/main/get-wikimedia.sh

LINK_REGEX = re.compile(r'\b(?:https?:\/\/|www\.|pic.)(?:[\w-]+\.)*[\w-]+(?:\.(?:[a-z]{2,3}))(?:/)?(?:\w+?)?\b')
EMAIL_REGEX = re.compile(r'\S+@\S+')
QUOTATION_MARKS = re.compile(r"([“”«»„‹›❝❞〝〞〟＂＇〈〉《》「」『』【】〔〕〖〗〘〙〚〛‚‛‟<>])")
PUNCTUATION = re.compile(r'([!"#$%&\'()*+,-.…/:;<=>?@\[\\\]^_`{|–}∙—~])')
NUMBERS = re.compile(r'\d+')
WHITESPACE = re.compile(r'\s{2,}')
LINE_BREAKS = re.compile(r'\n')
HTML_FRAGMENTS = re.compile(r'(?:<(a|br|p|span|bold) .*?>)|(?:<\/(a|br|p|span|bold)>)')

wrong_encoding = ["â€ž", "Ã¶", "Ã¼", "Ã¼", "Ã¤", "â€"]

if __name__ == "__main__":
    print('testing regex')

    t1 = "Amazon prüft „weitere Konsequenzen“.…"
    assert PUNCTUATION.sub(" ", t1) == "Amazon prüft „weitere Konsequenzen“  ", PUNCTUATION.sub(" ", t1)
    assert QUOTATION_MARKS.sub(" ", t1) == "Amazon prüft  weitere Konsequenzen .…", QUOTATION_MARKS.sub(" ", t1)

    t2 = "☎ 0 42 82/20 43, www.nassfeld.attopFIT\nRe∙Cycle"

    assert LINK_REGEX.sub(" ", t2) == "☎ 0 42 82/20 43,  \nRe∙Cycle", LINK_REGEX.sub(" ", t2)
    assert PUNCTUATION.sub(" ", t2) == "☎ 0 42 82 20 43  www nassfeld attopFIT\nRe Cycle", PUNCTUATION.sub(" ", t2)
    assert NUMBERS.sub(" ", t2) == "☎      /   , www.nassfeld.attopFIT\nRe∙Cycle"
    assert LINE_BREAKS.sub(" ", t2) == "☎ 0 42 82/20 43, www.nassfeld.attopFIT Re∙Cycle", LINE_BREAKS.sub(" ", t2)

    t3 = "www.respekt.net Initiativen für ein besseres Zusammenleben ausgezeichnet"

    assert LINK_REGEX.sub(" ", t3) == "  Initiativen für ein besseres Zusammenleben ausgezeichnet", LINK_REGEX.sub(" ", t3)

    t4 = "www.politik-live.at am Samstag berichtete, konfrontierten Ermittler des BAK Schüssel mit einem Chat zwischen Wolf und dem ehemaligen Generalsekretär Thomas Schmid, indem sich Wolf offenbar auf Schüssel bezog."

    assert LINK_REGEX.sub(" ", t4) == "  am Samstag berichtete, konfrontierten Ermittler des BAK Schüssel mit einem Chat zwischen Wolf und dem ehemaligen Generalsekretär Thomas Schmid, indem sich Wolf offenbar auf Schüssel bezog.", LINK_REGEX.sub(" ", t4)

    t5 = 'Die <a href="http://www.tripadvisor.at" target="_blank">Tripadvisor</a>-Community hat die besten österreichischen Museen gewählt'
    
    assert LINK_REGEX.sub(" ", t5) == 'Die <a href=" " target="_blank">Tripadvisor</a>-Community hat die besten österreichischen Museen gewählt', LINK_REGEX.sub(" ", t5)
    assert HTML_FRAGMENTS.sub(" ", t5) == 'Die  Tripadvisor -Community hat die besten österreichischen Museen gewählt', HTML_FRAGMENTS.sub(" ", t5)

    t6 = "anAragon: Tierschutzkompetenzzentrum Kärnten, Tel.: 0463/435 41 23,www.tiere-in-not.at."

    assert LINK_REGEX.sub(" ", t6) == "anAragon: Tierschutzkompetenzzentrum Kärnten, Tel.: 0463/435 41 23, .", LINK_REGEX.sub(" ", t6)

    t7 = "determined not to give up https://t.co/gocokspAv6 pic.twitter.com/glHt2dPeRA— The Daily Beast (@thedailybeast)"

    assert LINK_REGEX.sub(" ", t7) == "determined not to give up    — The Daily Beast (@thedailybeast)", LINK_REGEX.sub(" ", t7)
    assert PUNCTUATION.sub(" ", t7) == "determined not to give up https   t co gocokspAv6 pic twitter com glHt2dPeRA  The Daily Beast   thedailybeast ", PUNCTUATION.sub(" ", t7)

    t8 = "pic.twitter.com/30bpn0dAFN\n\n— UNHCR, the UN Refugee Agency (@Refugees)"

    assert LINK_REGEX.sub(" ", t8) == " \n\n— UNHCR, the UN Refugee Agency (@Refugees)", LINK_REGEX.sub(" ", t8)

    print('All tests passed!')