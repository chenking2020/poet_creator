import re


def extract_all_features(input_infos):
    all_features = []
    for poet in input_infos:
        sentences = re.split("，|。", poet)
        if len(sentences) == 0:
            continue
        if len(sentences[0]) != 5 and len(sentences[0]) != 7:
            continue
        effective_sentences = [sentences[0]]
        for i in range(1, len(sentences)):
            if len(sentences[i]) == len(effective_sentences[0]):
                effective_sentences.append(sentences[i])
        for sentence in effective_sentences:
            words = ["<START>"]
            for word in sentence:
                words.append(word)
            if len(words) == 0:
                continue
            words.append("<END>")
            labels = words[1:]
            all_features.append([words, labels])

    return all_features
