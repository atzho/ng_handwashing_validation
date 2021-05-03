import sys, pickle, torch, itertools, time, math
import numpy as np
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class HandWashDataset(Dataset):
    # time_window_len = time (in seconds) that the RNN remembers
    def __init__(self, mediapipe_pos_file, mediapipe_neg_file, time_window_len, transform=None, target_transform=None):
        with open(mediapipe_pos_file) as f:
            self.mediapipe_pos_data = json.load(f)
            #for video_arr in self.mediapipe_pos_data:
            #    video_arr['label'] = 1
        with open(mediapipe_neg_file) as f:
            self.mediapipe_neg_data = json.load(f)
            #for video_arr in self.mediapipe_neg_data:
            #    video_arr['label'] = 0
        self.transform = transform
        self.target_transform = target_transform

        self.time_window_len = time_window_len
        self.frame_window_len = 2 * self.time_window_len

        count = 0
        data = []
        labels = []
        #print(self.mediapipe_pos_data[0]['output'][0:6]) #debugging
        # cut positive training data into frames
        for video in self.mediapipe_pos_data:
            window_count = len(video['output']) - self.frame_window_len + 1
            if window_count > 0: #if contains enough frames
                for i in range(window_count):
                    data.append(np.array(video['output'][i:(i + self.frame_window_len)]))
                    #labels.append(video['label'])
                    labels.append(1)
                    count += 1
        
        # cut negative training data into frames
        for video in self.mediapipe_neg_data:
            window_count = len(video['output']) - self.frame_window_len + 1
            if window_count > 0: #if contains enough frames
                for i in range(window_count):
                    data.append(np.array(video['output'][i:(i + self.frame_window_len)]))
                    #labels.append(video['label'])
                    labels.append(0)
                    count += 1
        
        #print("Data shape:")
        #print(np.array(data[0]))
        self.data = np.stack(data, axis=0)
        self.labels = np.array(labels)
        self.len = count

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #print("Running get item")
        #print(self.data.shape)
        video_seq = self.data[idx, :, :, :]
        #print("Video seq. shape:")
        #print(video_seq.shape)
        video_label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #sample = {"MediaPipe Data": video_seq, "Label": video_label}
        sample = (video_seq, video_label)
        #print(sample)
        return sample

class HandWashNet(torch.nn.Module):

    def __init__(self, input_size=3, hidden_size=10, rnn_layers=6, dropout_probability=0.2):
        #Initialization
        super(HandWashNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout_probability = dropout_probability

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        #Layers
        self.rnn = torch.nn.RNN(input_size,
                                hidden_size,
                                rnn_layers,
                                nonlinearity='relu',
                                dropout=dropout_probability)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def init_hidden(self, batch_size):
        #hidden = torch.zeros(self.rnn_layers, batch_size, self.hidden_size).to(self.device)
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_size).to(self.device)

    def forward(self, x_batch, hidden):
        print("Entering forward")
        #print(x_batch.shape)
        #batch_size = x_batch.size(0)
        #print("Batch size:")
        #print(batch_size)
        #print(hidden.shape)

        out, hidden = self.rnn(x_batch, hidden)

        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        out = self.sigmoid(out)

        print("Leaving forward")
        return out, hidden

batch_size = 2

def train_loop(dataloader, model, loss_fn, optimizer):
    print("Entering training loop...")
    #print(model)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #print(X.shape)
        # Compute prediction and loss
        #hidden_state = model.init_hidden(batch_size)
        hidden_state = model.init_hidden(batch_size)
        for i in range(batch_size):
            pred, hidden_state = model(X[i,:,:,:].float(), hidden_state)
            #print("Prediction:")
            #print(pred[0].shape)
            #print(pred[1].shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if batch % 2 == 1:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    print("Entering testing loop...")
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# under construction - list of global vars for now

training_dataset = HandWashDataset("./test.json", "./test.json", 3)
#print(training_dataset.data.shape)
test_dataset = HandWashDataset("./test.json", "./test.json", 3)

#training_data = HandWashDataset("./preprocessed_positives.json", "./preprocessed_positives.json", 3)
#test_data = HandWashDataset("./preprocessed_positives.json", "./preprocessed_positives.json", 3)

labels_map = {
    0: "Not Handwashing",
    1: "Handwashing",
}

train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#print("loaded")
#print(train_dataloader)

#data, lbl = training_dataset.__getitem__(0)

#train_features, train_labels = next(iter(train_dataloader))
#print("Data: ")
#print(train_features)
#print("Label: ")
#print(train_labels)
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#video_seq = train_features[0]
#print(video_seq)
#label = train_labels[0]
#print(f"Label: {label}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model = HandWashNet().to(device)
model = model.float()
#print(model)

print("Model structure: ", model, "\n\n")
#for name, param in model.named_parameters():
#    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Hyperparameters
learning_rate = 1e-3
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 5

train_loop(train_dataloader, model, loss_fn, optimizer)

'''
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
torch.save(model.state_dict(), 'handwashnet_weights.pth')'''
print("Done!")