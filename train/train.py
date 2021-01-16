from __future__ import print_function

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import importlib
from write_provider.dataprocess import data_loader
import datetime
import json


class TrainProcess(object):
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(self.BASE_DIR, "params.json")) as f:
            config_list = json.load(f)
            for config_info in config_list:
                if config_info["active"] == "true":
                    self.params = config_info

    def train_process(self):

        print("{}\tstart reading data...".format(self.get_now_time()))
        model_path = os.path.join(self.BASE_DIR, "checkpoint", self.params["model_name"])
        data_loader.init_processor(self.params["corpus_name"])
        all_train_sentences = data_loader.load_data(
            os.path.join(self.BASE_DIR, "data", self.params["corpus_name"], "train.txt"))
        # all_dev_sentences = data_loader.load_data(os.path.join(data_path, model_tags["corpus_name"], "dev.txt"))

        word_to_id, id_to_word, py_to_id, id_to_py, ton_to_id, id_to_ton = data_loader.word_mapping(all_train_sentences,
                                                                                                    self.params[
                                                                                                        "word_pretrain_type"],
                                                                                                    os.path.join(
                                                                                                        self.params[
                                                                                                            "pretrain_basedir"],
                                                                                                        self.params[
                                                                                                            "word_pretrain_name"]))

        train_data = data_loader.prepare_dataset(all_train_sentences, word_to_id, py_to_id, ton_to_id,
                                                 self.params["word_pretrain_type"])
        # dev_data = data_loader.prepare_dataset(all_dev_sentences, word_to_id, algo_params["word_pretrain_type"])

        train_manager = data_loader.BatchManager(train_data, self.params["batch_size"])
        # dev_manager = data_loader.BatchManager(dev_data, int(algo_params["batch_size"]))

        # log_file.write("{}\tfinish reading data!, percent of train/dev: {}/{}\n".format(self.get_now_time(),
        #                                                                                 len(all_train_sentences),
        #                                                                                 len(all_dev_sentences)))
        print("{}\tfinish reading data!, percent of train/dev: {}/{}".format(self.get_now_time(),
                                                                             len(all_train_sentences), 0))

        # ToDo 这里核心是根据task_id读取到所用到的模型，灵活初始化模型
        print("{}\tstart init model...".format(self.get_now_time()))
        write_module = importlib.import_module("write_provider.model.{}".format(self.params["algo_name"]))
        write_model = write_module.WriteModule(self.params, len(word_to_id), len(py_to_id), len(ton_to_id))

        write_model.init_word_embedding(self.params["word_pretrain_type"],
                                        os.path.join(
                                            self.params["pretrain_basedir"],
                                            self.params[
                                                "word_pretrain_name"]))
        write_model.init_pinyin_embedding(self.params["word_pretrain_type"],
                                          os.path.join(
                                              self.params["pretrain_basedir"],
                                              self.params[
                                                  "word_pretrain_name"]))
        write_model.init_ton_embedding(self.params["word_pretrain_type"],
                                       os.path.join(
                                           self.params["pretrain_basedir"],
                                           self.params[
                                               "word_pretrain_name"]))
        write_model.set_optimizer()

        print("{}\tfinished init model!".format(self.get_now_time()))

        tot_length = len(all_train_sentences)

        print("{}\tstart training...".format(self.get_now_time()))
        for epoch_idx in range(self.params["epoch"]):

            epoch_loss = 0
            write_model.start_train_setting()
            iter_step = 0
            for batch in train_manager.iter_batch(shuffle=True):
                iter_step += 1
                _, loss = write_model.train_batch(batch)
                epoch_loss += loss
                print("{}\t{}".format(self.get_now_time(),
                                      "epoch: %s, current step: %s, current loss: %.4f" % (
                                          epoch_idx, iter_step, loss / len(batch))))

            epoch_loss /= tot_length

            write_model.end_train_setting()

            try:
                write_model.save_checkpoint({
                    'word_to_id': word_to_id,
                    'id_to_word': id_to_word,
                    "py_to_id": py_to_id,
                    "id_to_py": id_to_py,
                    "ton_to_id": ton_to_id,
                    "id_to_ton": id_to_ton
                }, os.path.join(model_path, "write"))
            except Exception as inst:
                print(inst)

    def get_now_time(self):
        timenow = (datetime.datetime.utcnow() + datetime.timedelta(hours=8))
        now_time_str = timenow.strftime("%Y-%m-%d %H:%M:%S")
        return now_time_str


if __name__ == '__main__':
    train_i = TrainProcess()
    train_i.train_process()
