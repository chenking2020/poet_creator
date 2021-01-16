from pypinyin import lazy_pinyin, Style

style = Style.FINALS_TONE3


def extract_all_features(input_infos):
    all_features = []
    for poet in input_infos:
        words = ["<START>"]
        for word in poet:
            words.append(word)
        words.append("<END>")

        pinyins = ["<START>"]
        tons = ["<START>"]
        for py in lazy_pinyin(poet, style=style):
            ton_num = py[-1]
            if ton_num in ["1", "2", "3", "4"]:
                pinyins.append(py[:-1])
                tons.append(ton_num)
            else:
                pinyins.append(py)
                tons.append("0")

        pinyins.append("<END>")
        tons.append("<END>")

        if len(words) != len(pinyins) or len(words) != len(pinyins):
            continue

        labels = words[1:]
        all_features.append([words, pinyins, tons, labels])

    return all_features
