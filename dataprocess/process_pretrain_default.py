def create_dico(item_list):
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences, pretrain_path):
    words = [[x.lower() for x in s[0]] for s in sentences]
    word_dico = create_dico(words)
    word_dico["<PAD>"] = 100000001
    word_dico['<UNK>'] = 100000000
    word_to_id, id_to_word = create_mapping(word_dico)

    return word_dico, word_to_id, id_to_word


def pinyin_mapping(sentences, pretrain_path):
    id = 2
    py_to_id = {"<PAD>": 0, "<UNK>": 1}
    id_to_py = {0: "<PAD>", 1: "<UNK>"}
    for s in sentences:
        for x in s[1]:
            if x.lower() not in py_to_id:
                py_to_id[x.lower()] = id
                id_to_py[id] = x.lower()
                id += 1
    return None, py_to_id, id_to_py


def ton_mapping(sentences, pretrain_path):
    id = 2
    ton_to_id = {"<PAD>": 0, "<UNK>": 1}
    id_to_ton = {0: "<PAD>", 1: "<UNK>"}
    for s in sentences:
        for x in s[2]:
            if x.lower() not in ton_to_id:
                ton_to_id[x.lower()] = id
                id_to_ton[id] = x.lower()
                id += 1
    return None, ton_to_id, id_to_ton


def sentence_to_word_seq(word_to_id, word_list):
    return [word_to_id[w.lower() if w.lower() in word_to_id else '<UNK>']
            for w in word_list]


def sentence_to_pinyin_seq(py_to_id, pinyin_list):
    return [py_to_id[w.lower() if w.lower() in py_to_id else '<UNK>']
            for w in pinyin_list]


def sentence_to_ton_seq(ton_to_id, ton_list):
    return [ton_to_id[w.lower() if w.lower() in ton_to_id else '<UNK>']
            for w in ton_list]
