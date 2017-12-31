import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.model          = kwargs["model"]
        self.batch_size     = kwargs["batch_size"]
        self.max_sent_len   = kwargs["max_sent_len"]
        self.sentence_num   = kwargs["sentence_num"]
        self.word_dim       = kwargs["word_dim"]
        self.vocab_size     = kwargs["vocab_size"]
        self.class_size     = kwargs["class_size"]
        self.filters        = kwargs["filters"] # filter windows = [3,4,5]
        self.filter_num     = kwargs["filter_num"] # 100 feature maps for each window size
        self.dropout_prob   = kwargs["dropout_prob"]
        self.in_channel     = 1
        self.pad_id         = 0 # word_id of <pad>

        assert (len(self.filters) == len(self.filter_num))

        # one for UNK (id=1) and one for zero padding (id=0)
        self.embedding = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=self.pad_id)
        if self.model == "static" or self.model == "non-static" or self.model == "multichannel":
            self.wv_matrix = kwargs["wv_matrix"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            if self.model == "static":
                self.embedding.weight.requires_grad = False
            elif self.model == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=self.pad_id)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2

        for i in range(len(self.filters)):
            conv = nn.Conv1d(in_channels=self.in_channel, out_channels=self.filter_num[i], kernel_size=self.word_dim * self.filters[i], stride=self.word_dim)
            setattr(self, 'conv_%d' % i , conv)

        self.fc = nn.Linear(self.sentence_num * sum(self.filter_num), self.class_size)

    def get_conv(self, i):
        return getattr(self, 'conv_%d' % i)

    def forward(self, input):
        # if x.size=[batch_size, max_sent_len], make it to [batch_size, 1, max_sent_len]
        if len(input.size()) == 2:
            input = input.unsqueeze(1)
        input = torch.transpose(input, 0, 1) # transpose to [sent_num, batch_size, max_sent_len]

        sent_x = []
        for k in range(input.size(0)):
            x = self.embedding(input[k]).view(-1, 1, self.word_dim * self.max_sent_len) # (batch_size, max_sent_len, emb_dim) -> (batch_size, 1, max_sent_len*emb_dim)
            if self.model == "multichannel":
                x2 = self.embedding2(input[k]).view(-1, 1, self.word_dim * self.max_sent_len)
                x = torch.cat((x, x2), 1) # [batch_size, 2, max_sent_len * emb_dim]

            conv_results = []
            for i in range(len(self.filters)):
                conv_result = F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_sent_len - self.filters[i] + 1).view(-1, self.filter_num[i]) # F.relu=(batch_size, out_channel, max_sent_len - window_size + 1)， F.max_pool1d=(batch_size, out_channel, 1) -> .view()=(batch_size, out_channel)
                conv_results.append(conv_result)

            x = torch.cat(conv_results, 1)
            sent_x.append(x)

        x = torch.cat(sent_x, 1)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.fc(x)

        return x
