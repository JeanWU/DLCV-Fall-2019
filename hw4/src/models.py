import torch.nn as nn
import torchvision.models as models
import torch

class Feature_Extractor(nn.Module):

    def __init__(self):
        super(Feature_Extractor, self).__init__()

        pretrained_model = models.resnet50(pretrained=True)
        #pretrained_model = models.densenet121(pretrained=True)
        self.model = nn.Sequential(
            *list(pretrained_model.children())[:-1],
        )

    def forward(self, img):
        out = self.model(img)   #torch.Size([batch, 2048, 4, 5])
        out = out.view(img.shape[0], -1)    #torch.Size([batch, 40960])
        return out


class FC(nn.Module):

    def __init__(self):
        super(FC, self).__init__()
        self.model = nn.Sequential(
            #nn.Linear(2048, 500),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 11),
        )


    def forward(self, feature):
        return self.model(feature)

def P1():
    return Feature_Extractor(), FC()


class BiRNN(nn.Module):
    def __init__(self, args):
        super(BiRNN, self).__init__()
        input_size = 2048
        num_classes = 11
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.FC = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size*2),
            nn.Linear(self.hidden_size*2, num_classes),
        )


    def forward(self, padded_sequence, input_lengths):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, padded_sequence.size(0), self.hidden_size).cuda() # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, padded_sequence.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths, batch_first=True)
        packed_out, (h_last, c_last) = self.lstm(packed, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        h_combined = torch.cat((h_last[1], h_last[3]), dim=1)
        out = self.FC(h_combined)

        return out, h_combined

def P2(args):
    return Feature_Extractor(), BiRNN(args)


class seq2seq(nn.Module):
    def __init__(self, args):
        super(seq2seq, self).__init__()
        input_size = 2048
        num_classes = 11
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.FC = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size*2),
            nn.Linear(self.hidden_size*2, num_classes),
        )


    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda() # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        # Forward propagate LSTM
        out, (h_last, c_last) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        out = self.FC(out.squeeze(0))   # out.squeeze(0) [batch #, hidden size*2]

        return out

def P3(args):
    return Feature_Extractor(), seq2seq(args)
