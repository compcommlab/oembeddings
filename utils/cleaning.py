import re
import emoji

# regular expressions to clean raw sentences.
# We follow the procedures that the Facebook engineers used for fasttext:
# References:
# https://github.com/facebookresearch/fastText/blob/main/tests/fetch_test_data.sh
# https://github.com/facebookresearch/fastText/blob/main/get-wikimedia.sh

LINK_REGEX = re.compile(r'\b(?:https?:\/\/|www\.|pic.)(?:[\w-]+\.)*[\w-]+(?:\.(?:[a-z]{2,3}))(?:/)?(?:\w+?)?\b')
EMAIL_REGEX = re.compile(r'\S+@\S+')
QUOTATION_MARKS = re.compile(r"([“”«»„‹›❝❞〝〞〟＂＇〈〉《》「」『』【】〔〕〖〗〘〙〚〛‚‛‟<>])")
PUNCTUATION = re.compile(r'([!"#$%&\'()+,.…/;<=>?@\[\\\]^`{|–}∙—~])')
PUNCTUATION_ALL = re.compile(r'([!"#$%&\'()*+,-.…/:;<=>?@\[\\\]^_`{|–}∙—~])')
NUMBERS = re.compile(r'\d+')
WHITESPACE = re.compile(r'\s{2,}')
LINE_BREAKS = re.compile(r'\n')
HTML_FRAGMENTS = re.compile(r'(?:<(a|br|p|span|bold) .*?>)|(?:<\/(a|br|p|span|bold)>)')
GENDERSTAR = re.compile(r'(\w+)([\*\:\_])([Ii]nnen\w*)')
STAR_COLON_UNDERSCORE = re.compile(r'([*:]|\b_)')

# preserve some hyphenated words: E-Mail stays, "FPÖ-Wähler" replaced with whitespace "FPÖ Wähler"
HYPHENATED = re.compile(r'\b(\w{2,})-+(\w+)') 

# remove hyphens at the end of words: "Zu- und Ablauf"; at the beginning: "-word"; or isolated " - "
HYPHEN_SUFFIX = re.compile(r'(\w+)-+(\s|$)') 
HYPHEN_PREFIX = re.compile(r'(^|\s)-+(\w+)') 
HYPHEN_ISOLATED = re.compile(r'(^|\W)-+(\W|$)')

wrong_encoding = ["â€ž", "Ã¶", "Ã¼", "Ã¼", "Ã¤", "â€"]



def clean_sentence(text: str,
                     lowercase=False,
                     remove_links=True,
                     remove_emails=True,
                     remove_emojis=True,
                     remove_punctuation=True,
                     remove_numbers=True,
                     remove_quotations=False,
                     genderstar=True) -> str:
    
    text = HTML_FRAGMENTS.sub(" ", text)
    
    if lowercase:
        text = text.lower()
    if remove_links:
        text = LINK_REGEX.sub(" ", text)
    if remove_emails:
        text = EMAIL_REGEX.sub(" ", text)

    if genderstar:
        # preserve genderstar (normalize with underscore)
        text = GENDERSTAR.sub(r"\1_\3", text)
    
    if remove_emojis:
        text = emoji.replace_emoji(text, " ")
    if remove_punctuation:
        if genderstar:
            text = PUNCTUATION.sub(" ", text)
        else:
            text = PUNCTUATION_ALL.sub(" ", text)
        
    else:
        # pad punctuation with whitespace
        text = PUNCTUATION.sub(r' \1 ', text)
    if remove_quotations:
        text = QUOTATION_MARKS.sub(" ", text)
    else:
        # pad quotation marks and normalize
        text = QUOTATION_MARKS.sub(r' " ', text)

    if remove_numbers:
        text = NUMBERS.sub("", text)
    
    if genderstar:
        # another pass at the end; remove other forms of star/colons
        text = STAR_COLON_UNDERSCORE.sub("", text)

    # handle hyphenated words
    # preserve: "E-Mail" -> "E-Mail"
    # otherwise remove hyphens: "EU-Beitritt" -> "EU Beitritt"
    text = HYPHENATED.sub(r"\1 \2", text)
    text = HYPHEN_PREFIX.sub(r" \2 ", text)
    text = HYPHEN_SUFFIX.sub(r"\1 ", text)
    text = HYPHEN_ISOLATED.sub(r" ", text)
    text = HYPHENATED.sub(r"\1 \2", text)
    text = HYPHENATED.sub(r"\1 \2", text)

    
    text = LINE_BREAKS.sub(" ", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()
    

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
    assert PUNCTUATION.sub(" ", t7) == "determined not to give up https:  t co gocokspAv6 pic twitter com glHt2dPeRA  The Daily Beast   thedailybeast ", PUNCTUATION.sub(" ", t7)

    t8 = "pic.twitter.com/30bpn0dAFN\n\n— UNHCR, the UN Refugee Agency (@Refugees)"

    assert LINK_REGEX.sub(" ", t8) == " \n\n— UNHCR, the UN Refugee Agency (@Refugees)", LINK_REGEX.sub(" ", t8)

    t9 = 'Das Motto der Aktion: "Demokrat*innen nicht hängen lassen".'
    t10 = 'In unterschiedlichen Wiener Clubs und Musikspielstätten werden Auftritte von Künstler*innen und DJ-Sets live'
    t11 = 'Kritiker:innen weisen darauf hin, dass keinerlei Beleg für die angebliche Wirksamkeit existiert.'
    t12 = 'Das Motto der Aktion: "Demokrat_innen nicht hängen lassen".'

    assert GENDERSTAR.sub(r"\1_\3", t9) == 'Das Motto der Aktion: "Demokrat_innen nicht hängen lassen".', GENDERSTAR.sub(r"\1\_\3", t9)
    assert GENDERSTAR.sub(r"\1_\3", t10) == 'In unterschiedlichen Wiener Clubs und Musikspielstätten werden Auftritte von Künstler_innen und DJ-Sets live', GENDERSTAR.sub(r"\1\_\3", t10)
    assert GENDERSTAR.sub(r"\1_\3", t11) == 'Kritiker_innen weisen darauf hin, dass keinerlei Beleg für die angebliche Wirksamkeit existiert.', GENDERSTAR.sub(r"\1\_\3", t11)
    assert GENDERSTAR.sub(r"\1_\3", t12) == 'Das Motto der Aktion: "Demokrat_innen nicht hängen lassen".', GENDERSTAR.sub(r"\1\_\3", t12)
    
    # sentences where we don't want special characters; but also want to preserve genderstar
    t13 = '* "Fußball für Vielfalt" hieß noch bis vor kurzem "Fußball gegen Homophobie" und gehört zur Bundesstiftung Magnus Hirschfeld.'
    t14 = '"Es ist Bahnstreik - sche***egal!".'
    assert GENDERSTAR.sub(r"\1_\3", t13) == '* "Fußball für Vielfalt" hieß noch bis vor kurzem "Fußball gegen Homophobie" und gehört zur Bundesstiftung Magnus Hirschfeld.', GENDERSTAR.sub(r"\1\_\3", t13)
    assert GENDERSTAR.sub(r"\1_\3", t14) == '"Es ist Bahnstreik - sche***egal!".', GENDERSTAR.sub(r"\1\_\3", t14)


    # stacked cleaning: genderstar, punctuation, quotation marks, remaining special characters
    t9_a = GENDERSTAR.sub(r"\1_\3", t9)
    t9_b = PUNCTUATION.sub(" ", t9_a)
    t9_c = QUOTATION_MARKS.sub(" ", t9_b)
    t9_d = STAR_COLON_UNDERSCORE.sub(" ", t9_c)

    assert t9_d == 'Das Motto der Aktion   Demokrat_innen nicht hängen lassen  ', t9_d

    assert clean_sentence(t9) == 'Das Motto der Aktion Demokrat_innen nicht hängen lassen',  clean_sentence(t9)

    t10_a = GENDERSTAR.sub(r"\1_\3", t10)
    t10_b = PUNCTUATION.sub(" ", t10_a)
    t10_c = QUOTATION_MARKS.sub(" ", t10_b)
    t10_d = STAR_COLON_UNDERSCORE.sub(" ", t10_c)
    t10_e = HYPHENATED.sub(r"\1 \2", t10_d)

    assert t10_e == 'In unterschiedlichen Wiener Clubs und Musikspielstätten werden Auftritte von Künstler_innen und DJ Sets live', t10_e


    t11_a = GENDERSTAR.sub(r"\1_\3", t11)
    t11_b = PUNCTUATION.sub(" ", t11_a)
    t11_c = QUOTATION_MARKS.sub(" ", t11_b)
    t11_d = STAR_COLON_UNDERSCORE.sub(" ", t11_c)

    assert t11_d == 'Kritiker_innen weisen darauf hin  dass keinerlei Beleg für die angebliche Wirksamkeit existiert ', t11_d

    t13_a = GENDERSTAR.sub(r"\1_\3", t13)
    t13_b = PUNCTUATION.sub(" ", t13_a)
    t13_c = QUOTATION_MARKS.sub(" ", t13_b)
    t13_d = STAR_COLON_UNDERSCORE.sub(" ", t13_c)

    assert t13_d == '   Fußball für Vielfalt  hieß noch bis vor kurzem  Fußball gegen Homophobie  und gehört zur Bundesstiftung Magnus Hirschfeld ', t13_d

    t14_a = GENDERSTAR.sub(r"\1_\3", t14)
    t14_b = PUNCTUATION.sub(" ", t14_a)
    t14_c = QUOTATION_MARKS.sub(" ", t14_b)
    t14_d = STAR_COLON_UNDERSCORE.sub(" ", t14_c)
    t14_e = HYPHENATED.sub(r"\1 \2", t14_d)
    t14_f = HYPHEN_PREFIX.sub(r" \1 ", t14_e)
    t14_g = HYPHEN_SUFFIX.sub(r"\1 ", t14_f)
    t14_h = HYPHEN_ISOLATED.sub(r" ", t14_g)

    assert t14_h == ' Es ist Bahnstreik sche   egal   ', t14_h

    t15 = "EU-Austritt E-Mobilität U-Ausschuss FPÖ-Wähler ACLU-Anwalt U-Kommission BVT-U-Ausschuss „Ibiza“-U-Ausschuss E-Mail -partikel Ab- und Zulauf."

    t15_a = HYPHENATED.sub(r"\1 \2", t15)
    assert t15_a == "EU Austritt E-Mobilität U-Ausschuss FPÖ Wähler ACLU Anwalt U-Kommission BVT U-Ausschuss „Ibiza“-U-Ausschuss E-Mail -partikel Ab- und Zulauf.", t15_a

    t15_b = HYPHEN_PREFIX.sub(r" \2", t15_a)
    t15_c = HYPHEN_SUFFIX.sub(r"\1 ", t15_b)
    t15_d = HYPHEN_ISOLATED.sub(r" ", t15_c)
    assert t15_d == "EU Austritt E-Mobilität U-Ausschuss FPÖ Wähler ACLU Anwalt U-Kommission BVT U-Ausschuss „Ibiza“-U-Ausschuss E-Mail partikel Ab und Zulauf.", t15_d

    t16 = "Einen Tag nach Rang vier vom 1-m-Brett hat sich"

    print('All tests passed!')