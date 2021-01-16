from __future__ import print_function

import os
import sys
from pypinyin import lazy_pinyin, Style

style = Style.FINALS_TONE3

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from write_provider.dataprocess import data_loader
import json
import importlib

id_params_map = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(os.path.join(BASE_DIR, "params.json")) as f:
    config_list = json.load(f)
    for config_info in config_list:
        if config_info["active"] == "true":
            id_params_map[config_info["model_id"]] = config_info


class StartWordWriter(object):
    def __init__(self, model_id):
        self.params = id_params_map[model_id]
        self.processor_module = importlib.import_module(
            'write_provider.dataprocess.process_{}'.format(self.params["corpus_name"]))

        model_path = os.path.join(BASE_DIR, "checkpoint", self.params["model_name"])
        map_path = os.path.join(model_path, "write_map.json")
        model_file_path = os.path.join(model_path, "write.model")

        self.params["batch_size"] = 1
        with open(map_path, "r") as f:
            self.write_map = json.load(f)

        # self.params["batch_size"] = 1

        write_module = importlib.import_module(
            "write_provider.model.{}".format(self.params["algo_name"]))
        self.write_model = write_module.WriteModule(self.params, len(self.write_map["word_to_id"]),
                                                    len(self.write_map["py_to_id"]), len(self.write_map["ton_to_id"]))

        self.write_model.load_checkpoint_file(model_file_path)

    def get_generate_result(self, input_json):
        result_text = ""
        start_words = list(input_json["start_word"])
        prefix_word = "<START>"
        end_word = "<END>"
        hidden = None
        for i in range(input_json["word_len"]):

            pinyins = []
            tons = []
            for py in lazy_pinyin(prefix_word, style=style):
                ton_num = py[-1]
                if ton_num in ["1", "2", "3", "4"]:
                    pinyins.append(py[:-1])
                    tons.append(ton_num)
                else:
                    pinyins.append(py)
                    tons.append("0")

            dataset = data_loader.prepare_dataset([[[prefix_word], pinyins, tons, [end_word]]],
                                                  self.write_map["word_to_id"], self.write_map["py_to_id"],
                                                  self.write_map["ton_to_id"], self.params["word_pretrain_type"],
                                                  False)
            dataset_loader = data_loader.BatchManager(dataset, self.params["batch_size"], False)
            for batch in dataset_loader.iter_batch(shuffle=False):
                gen_id, hidden = self.write_model.predict_one(batch, hidden)
                if i < len(start_words):
                    result_text += start_words[i]
                    prefix_word = start_words[i]
                else:
                    gen_word = self.write_map["id_to_word"][str(gen_id)]
                    result_text += gen_word
                    prefix_word = gen_word
        return result_text


if __name__ == '__main__':
    w_model = StartWordWriter("1")
    result = w_model.get_generate_result({"start_word": "雨", "word_len": 24})
    print(result)
