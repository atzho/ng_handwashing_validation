import sys, numpy, pickle, torch, itertools, time, math

class HandWashNet(torch.nn.Module):

    def __init__(self, input_size=42, hidden_size=10, n_layers=2, dropout_probability=0.2):
        #Initialization
        super(HandWashNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_probability = dropout_probability

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        #Layers
        self.rnn = torch.nn.RNN(input_size,
                                hidden_size,
                                n_layers,
                                nonlinearity='relu',
                                dropout=dropout_probability)
        self.fc = torch.nn.Linear(hidden_size, 1)

        def forward(self, x_batch):
            batch_size = x_batch.size(0)
            hidden = self.init_hidden(batch_size)

            out, hidden = self.rnn(x_batch, hidden)

            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)

            return out, hidden

        def init_hidden(self, batch_size):
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
            return hidden
