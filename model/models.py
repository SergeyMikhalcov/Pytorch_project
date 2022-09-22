import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models

class VGG16(BaseModel):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
    
class Block(BaseModel):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class VGG16_pretrained(BaseModel):
    def __init__(self, num_classes=2):
        super(VGG16_pretrained, self).__init__()
        self.model = models.vgg16(pretrained=True)
        fc_inputs = self.model.classifier[0].in_features
        self.model.classifier = nn.Identity()
        self.classifier_tr = nn.Sequential(
            nn.Linear(fc_inputs, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),                      
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.model(images)
        labels = self.classifier_tr(features)
        return F.log_softmax(labels, dim=1) 
    
class ResNet_18(BaseModel):
    def __init__(self, image_channels=3, num_classes=2):
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
    
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
  
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
    
        return F.log_softmax(out, dim=1) 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
        
class ResNet_18_pretrained(BaseModel):
    def __init__(self, num_classes=2):
        super(ResNet_18_pretrained, self).__init__()
        self.model = models.resnet18(pretrained=True)
        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(fc_inputs, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),                      
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.model(images)
        labels = self.classifier(features)
        return F.log_softmax(labels, dim=1)
    
class SimpleRNN(BaseModel):
    def __init__(self, dictionary_size=95, embedding_size=190, num_hiddens=380, num_classes=95):
        super(SimpleRNN, self).__init__()
        self.num_classes = num_classes
        self.embedding = torch.nn.Embedding(dictionary_size, embedding_size)
        self.hidden = torch.nn.RNN(embedding_size, num_hiddens, batch_first=True)
        self.output = torch.nn.Linear(num_hiddens, num_classes)

    def forward(self, X):
        out = self.embedding(X)
        rnn_out, state = self.hidden(out)  
        predictions = self.output(rnn_out)
        
        return predictions

class GRU_RNN(BaseModel):
    def __init__(self, dictionary_size, embedding_size, num_hiddens, num_classes):
        super(GRU_RNN, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.embedding = torch.nn.Embedding(dictionary_size, embedding_size)
        self.hidden = torch.nn.GRU(embedding_size, num_hiddens, batch_first=True)
        self.output = torch.nn.Linear(num_hiddens, num_classes)

    def forward(self, X):
        out = self.embedding(X)
        rnn_out, state = self.hidden(out)  
        predictions = self.output(rnn_out)
        
        return predictions
    
class LSTM_RNN(BaseModel):
    def __init__(self, dictionary_size, embedding_size, num_hiddens, num_classes):
        super(LSTM_RNN, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.embedding = torch.nn.Embedding(dictionary_size, embedding_size)
        self.hidden = torch.nn.LSTM(embedding_size, num_hiddens, batch_first=True)
        self.output = torch.nn.Linear(num_hiddens, num_classes)

    def forward(self, X):
        out = self.embedding(X)
        rnn_out, state = self.hidden(out)  
        predictions = self.output(rnn_out)
        
        return predictions

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0])) 
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor):
        enc_hid_in = self.encoder.initHidden()
        
        enc_out, enc_hid = self.encoder.forward(input_tensor, enc_hid_in)
        dec_out, dec_hid = self.decoder.forward(enc_out, enc_hid)
        
        return dec_out, dec_hid