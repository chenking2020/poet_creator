import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.autograd as autograd
import json


class WriteModule(nn.Module):
    def __init__(self, params, vocab_size, pinyin_size, ton_size):

        super(WriteModule, self).__init__()
        self.params = params

        self.word_embeds = nn.Embedding(vocab_size, self.params["word_dim"])
        self.pinyin_embeds = nn.Embedding(pinyin_size, self.params["py_dim"])
        self.ton_embeds = nn.Embedding(ton_size, self.params["ton_dim"])

        self.word_lstm = nn.LSTM(self.params["word_dim"] + self.params["py_dim"] + self.params["ton_dim"],
                                 self.params["word_hidden"], num_layers=self.params["word_layers"], batch_first=True)

        self.l_model = nn.Linear(self.params["word_hidden"], vocab_size)

        self.word_rnn_layers = self.params["word_layers"]

        self.dropout = nn.Dropout(p=self.params["drop_out"])
        self.criterion = nn.CrossEntropyLoss()

        if self.params["gpu"] >= 0:
            torch.cuda.set_device(self.params["gpu"])
            self.word_embeds = self.word_embeds.cuda()
            self.pinyin_embeds = self.pinyin_embeds.cuda()
            self.ton_embeds = self.ton_embeds.cuda()
            self.word_lstm = self.word_lstm.cuda()
            self.dropout = self.dropout.cuda()

        self.batch_size = 1
        self.word_seq_length = 1

    def set_batch_seq_size(self, sentence):
        tmp = sentence.size()
        self.batch_size = tmp[0]
        self.word_seq_length = tmp[1]

    def init_word_embedding(self, pretrain_type, pretrain_path):
        if pretrain_type == "word_vector":
            with open(pretrain_path, "r") as f:
                vector_info = f.readline()
                vector_info = vector_info.strip().split()
                weights = torch.zeros(size=[int(vector_info[0]) + 2, int(vector_info[1])], dtype=torch.float)
                weights[0] = torch.rand(size=[int(vector_info[1])])
                weights[1] = torch.rand(size=[int(vector_info[1])])
                idx = 2
                for line in f:
                    line = line.strip().split()
                    weights[idx] = torch.tensor([float(v) for v in line[1:]])
                    idx += 1

            if self.params["gpu"] >= 0:
                self.word_embeds.weight = nn.Parameter(weights.cuda())
            else:
                self.word_embeds.weight = nn.Parameter(weights)

        else:
            nn.init.uniform_(self.word_embeds.weight, -0.25, 0.25)

    def init_pinyin_embedding(self, pretrain_type, pretrain_path):
        if pretrain_type == "word_vector":
            with open(pretrain_path, "r") as f:
                vector_info = f.readline()
                vector_info = vector_info.strip().split()
                weights = torch.zeros(size=[int(vector_info[0]) + 2, int(vector_info[1])], dtype=torch.float)
                weights[0] = torch.rand(size=[int(vector_info[1])])
                weights[1] = torch.rand(size=[int(vector_info[1])])
                idx = 2
                for line in f:
                    line = line.strip().split()
                    weights[idx] = torch.tensor([float(v) for v in line[1:]])
                    idx += 1

            if self.params["gpu"] >= 0:
                self.word_embeds.weight = nn.Parameter(weights.cuda())
            else:
                self.word_embeds.weight = nn.Parameter(weights)

        else:
            nn.init.uniform_(self.pinyin_embeds.weight, -0.25, 0.25)

    def init_ton_embedding(self, pretrain_type, pretrain_path):
        if pretrain_type == "word_vector":
            with open(pretrain_path, "r") as f:
                vector_info = f.readline()
                vector_info = vector_info.strip().split()
                weights = torch.zeros(size=[int(vector_info[0]) + 2, int(vector_info[1])], dtype=torch.float)
                weights[0] = torch.rand(size=[int(vector_info[1])])
                weights[1] = torch.rand(size=[int(vector_info[1])])
                idx = 2
                for line in f:
                    line = line.strip().split()
                    weights[idx] = torch.tensor([float(v) for v in line[1:]])
                    idx += 1

            if self.params["gpu"] >= 0:
                self.word_embeds.weight = nn.Parameter(weights.cuda())
            else:
                self.word_embeds.weight = nn.Parameter(weights)

        else:
            nn.init.uniform_(self.ton_embeds.weight, -0.25, 0.25)

    def set_optimizer(self):
        if self.params["update"] == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"])
        elif self.params["update"] == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"])

    def start_train_setting(self):
        self.train()

    def train_batch(self, batch):
        w_f, p_f, t_f, tg_v = batch
        self.zero_grad()
        out_put, loss, _ = self.forward(w_f, p_f, t_f, tg_v)
        epoch_loss = self.to_scalar(loss)
        loss.backward()
        self.clip_grad_norm()
        self.optimizer.step()
        return out_put, epoch_loss

    def end_train_setting(self):
        self.params["lr"] *= self.params["lr_decay"]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.params["lr"]
        return self.params["lr"]

    def eval_batch(self, batch):
        ws, ps, ts, tg = batch
        out_put, loss, _ = self.forward(ws, ps, ts, tg)
        return out_put, loss

    def predict_one(self, batch, hidden=None):
        ws, ps, ts, tg = batch
        tg = None
        out_put, loss, hidden = self.forward(ws, ps, ts, tg, hidden)
        max_index = torch.argmax(out_put, -1)
        return max_index.data.tolist()[0], hidden

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.params["clip_grad"])

    def to_scalar(self, var):
        return var.view(-1).data.tolist()[0]

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, data_map, filename):
        with open(filename + '_map.json', 'w') as f:
            f.write(json.dumps(data_map, ensure_ascii=False))
        torch.save(self.state_dict(), filename + '.model')

    def load_checkpoint_file(self, model_path):
        checkpoint_file = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint_file)
        self.eval()

    def forward(self, word_seq, py_seq, ton_seq, target_seq, hidden=None):

        word_seq = torch.LongTensor(word_seq)
        py_seq = torch.LongTensor(py_seq)
        ton_seq = torch.LongTensor(ton_seq)

        if target_seq is not None:
            target_seq = torch.LongTensor(target_seq)
        if self.params["gpu"] >= 0:
            word_seq = autograd.Variable(word_seq).cuda()
            py_seq = autograd.Variable(py_seq).cuda()
            ton_seq = autograd.Variable(ton_seq).cuda()
            if target_seq is not None:
                target_seq = autograd.Variable(target_seq).cuda()
        else:
            word_seq = autograd.Variable(word_seq)
            py_seq = autograd.Variable(py_seq)
            ton_seq = autograd.Variable(ton_seq)
            if target_seq is not None:
                target_seq = autograd.Variable(target_seq)

        self.set_batch_seq_size(word_seq)

        # hidden
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            h_0 = word_seq.data.new(2, self.batch_size, self.params["word_hidden"]).fill_(0).float()
            c_0 = word_seq.data.new(2, self.batch_size, self.params["word_hidden"]).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # word
        word_emb = self.word_embeds(word_seq)
        pinyin_emb = self.pinyin_embeds(py_seq)
        ton_emb = self.ton_embeds(ton_seq)
        # d_word_emb = self.dropout(word_emb)
        combine_emb = torch.cat((word_emb, pinyin_emb, ton_emb), dim=2)
        # combine
        # word level lstm
        lstm_out, hidden = self.word_lstm(self.dropout(combine_emb), (h_0, c_0))
        # d_lstm_out = self.dropout(lstm_out)
        out_put = self.l_model(lstm_out.contiguous().view(self.batch_size * self.word_seq_length, -1))
        # loss
        if target_seq is not None:
            loss = self.criterion(out_put, target_seq.view(-1))
        else:
            loss = None

        return out_put, loss, hidden
