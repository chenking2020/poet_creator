def extract_all_features(input_infos):
    all_features = []
    for poet in input_infos:
        words = ["<START>"]
        for word in poet:
            words.append(word)
        if len(words) == 0:
            continue
        words.append("<END>")
        labels = words[1:]
        all_features.append([words, labels])

    return all_features
