import torch
from torch import nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):    
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)
        self.bn = nn.BatchNorm1d(num_features=n_outputs)
    def forward(self, x):
        # x = self.bn(x)
        x = self.linear(x)
        x = self.bn(x)
        y_pred = torch.sigmoid(x)
        return y_pred
    

# class CnnLstmNet(nn.Module):
#     def __init__(self, output_class, doc: bool = False):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=2) 
#         self.batch_nor1 = nn.BatchNorm1d(num_features=64)
#         self.maxpool1 = nn.MaxPool1d(kernel_size=5, stride=1)  
#         self.dropout1 = nn.Dropout(p=0.5)
        
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2) 
#         self.batch_nor2 = nn.BatchNorm1d(num_features=128)
#         self.maxpool2 = nn.MaxPool1d(kernel_size=5, stride=1) 
#         self.dropout2 = nn.Dropout(p=0.5)

#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2) 
#         self.batch_nor3 = nn.BatchNorm1d(num_features=256)
#         self.maxpool3 = nn.MaxPool1d(kernel_size=5, stride=1) 
#         self.dropout3 = nn.Dropout(p=0.5)

#         self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2) 
#         self.batch_nor4 = nn.BatchNorm1d(num_features=512)
#         self.maxpool4 = nn.MaxPool1d(kernel_size=5, stride=1) 
#         self.dropout4 = nn.Dropout(p=0.5)

#         self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5, stride=2) 
#         self.batch_nor5 = nn.BatchNorm1d(num_features=1024)
#         self.maxpool5 = nn.MaxPool1d(kernel_size=5, stride=1) 
#         self.dropout5 = nn.Dropout(p=0.5)

#         self.hidden_size = 32
#         self.output_class = output_class
#         self.lstm1 = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=2, bidirectional=True, batch_first=True)
#         self.fc1 = nn.Linear(self.hidden_size*2, 50)
#         self.fc2 = nn.Linear(50, self.output_class)

#         self.doc = doc
#         self.logits = nn.ModuleList([LogisticRegression(1, 1) for i in range(self.output_class)])
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch_nor1(x)
#         x = F.relu(x)
#         x = self.maxpool1(x) 
#         x = self.dropout1(x) 
#         # print(x.shape)

#         x = self.conv2(x)
#         x = self.batch_nor2(x)
#         x = F.relu(x)
#         x = self.maxpool2(x) 
#         x = self.dropout2(x) 
#         # print(x.shape)

#         x = self.conv3(x)
#         x = self.batch_nor3(x)
#         x = F.relu(x)
#         x = self.maxpool3(x) 
#         x = self.dropout3(x) 
#         # print(x.shape)

#         x = self.conv4(x)
#         x = self.batch_nor4(x)
#         x = F.relu(x)
#         x = self.maxpool4(x) 
#         x = self.dropout4(x) 
#         # print(x.shape)

#         x = self.conv5(x)
#         x = self.batch_nor5(x)
#         x = F.relu(x)
#         x = self.maxpool5(x) 
#         x = self.dropout5(x) 
#         # print(x.shape)

#         x = x.transpose(1,2)
#         # print(x.shape)
#         x, (hn, cn) = self.lstm1(x)
#         x = x[:,-1,:]
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         if self.doc:
#             x = [i.unsqueeze(1) for i in torch.unbind(x, dim=1)]
#             xs = []
#             for indx, one_x in enumerate(x):
#                 one_y = self.logits[indx](one_x)
#                 xs.append(one_y)
#             x = torch.cat(xs, dim=1)

#         return x
    


class CnnLstmNet(nn.Module):
    def __init__(self, output_class, doc: bool = False):
        super().__init__()
        self.output_class = output_class

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv6 = nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.hidden_size = 32
        self.lstm1 = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size*2, 50)
        self.fc2 = nn.Linear(50, self.output_class)

        self.doc = doc
        self.logits = nn.ModuleList([LogisticRegression(1, 1) for i in range(self.output_class)])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        # print(x.shape)

        x = x.transpose(1,2)
        # print(x.shape)
        x, (hn, cn) = self.lstm1(x)
        x = x[:,-1,:]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        if self.doc:
            x = [i.unsqueeze(1) for i in torch.unbind(x, dim=1)]
            xs = []
            for indx, one_x in enumerate(x):
                one_y = self.logits[indx](one_x)
                xs.append(one_y)
            x = torch.cat(xs, dim=1)

        return x



class CNN1D(nn.Module):
    def __init__(self, num_classes=None, doc:bool = False):
        super(CNN1D, self).__init__()
        self.output_class = num_classes

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv6 = nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=1)


        self._residual_stack = ResidualStack(in_channels=1024,
                                             out_channels=1024,
                                             num_residual_layers=6,
                                             hidden_channels=1024)

        self.drop0 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(14336, 256) 
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(256, 256)
        self.drop1_2 = nn.Dropout(p=0.5)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc2 = None if num_classes is None else nn.Linear(256, num_classes)


        # self.drop = nn.Dropout(p=0.5)
        # self.fc1 = nn.Linear(14336, 256) # 1000
        # self.fc_relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = None if num_classes is None else nn.Linear(64, num_classes)

        self.doc = doc
        self.logits = nn.ModuleList([LogisticRegression(1, 1) for i in range(self.output_class)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        x = self._residual_stack(x)

        out = torch.flatten(x, start_dim=1)
        out = self.drop0(out)
        out = self.fc1(out)
        out = self.drop1(out)
        out = F.relu(out)
        out = self.fc1_2(out)
        out = self.drop1_2(out)
        if self.fc2 is not None:
            out = F.relu(out)
            out = self.fc2(out)

        # out = torch.flatten(x, start_dim=1)
        # out = self.drop(out)
        # out = self.fc1(out)
        # out = self.drop(out)
        # out = self.fc2(out)
        # out = self.drop(out)
        # if self.fc3 is not None:
        #     # out = self.fc_relu(out)
        #     out = self.fc3(out)


        if self.doc:
            x = [i.unsqueeze(1) for i in torch.unbind(out, dim=1)]
            xs = []
            for indx, one_x in enumerate(x):
                one_y = self.logits[indx](one_x)
                xs.append(one_y)
            out = torch.cat(xs, dim=1)

        return out


class CNN1D_Stft(nn.Module):
    def __init__(self, num_classes=None, doc:bool = False):
        super(CNN1D_Stft, self).__init__()
        self.output_class = num_classes

        self.conv1 = nn.Conv1d(101, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(11776, 256)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc_relu = nn.ReLU()
        self.fc2 = None if num_classes is None else nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # print(x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print(x.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        # print(x.shape)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        out = torch.flatten(x, start_dim=1)
        out = self.drop1(out)
        # print(f'out shape: {out.shape}')
        out = self.fc1(out)
        out = self.drop2(out)
        if self.fc2 is not None:
            # out = self.fc_relu(out)
            out = self.fc2(out)
        
        return out


class LSTM_Stft(nn.Module):
    def __init__(self, num_classes=None, doc:bool = False):
        super(LSTM_Stft, self).__init__()
        self.output_class = num_classes
        self.hidden_size = 200

        self.lstm1 = nn.LSTM(input_size=101, hidden_size=self.hidden_size, num_layers=2, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=101, hidden_size=self.hidden_size, num_layers=2, bidirectional=False, batch_first=True)

        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.hidden_size*2, 256)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc_relu = nn.ReLU()
        self.fc2 = None if self.output_class is None else nn.Linear(256, self.output_class)

    def forward(self, x):

        x = x.transpose(1,2)
        x1, (hn, cn) = self.lstm1(x)
        x2, (hn, cn) = self.lstm2(x)
        x = torch.cat([x1,x2], dim=2)
        x = x[:,-1,:]
        x = self.fc1(x)
        x = self.fc2(x)
    
        return x




class VAE(nn.Module):
    '''modified from https://blog.csdn.net/HYJ15679729212/article/details/132242694
    '''
    def __init__(self, input_dim: int = 1000, latent_dim: list = [512, 256, 128, 50]):
        super(VAE, self).__init__()
        # encoder
        self.encode_layers = nn.Sequential(
            nn.Linear(input_dim, latent_dim[0]),
            nn.ReLU(),
            nn.Linear(latent_dim[0], latent_dim[1]),
            nn.ReLU(),
            nn.Linear(latent_dim[1], latent_dim[2]),
            nn.ReLU(),
        )
        self.mean = nn.Linear(latent_dim[2], latent_dim[3])
        self.log_var = nn.Linear(latent_dim[2], latent_dim[3])
        # decoder
        self.decode_layers = nn.Sequential(
            nn.Linear(latent_dim[3], latent_dim[2]),
            nn.ReLU(),
            nn.Linear(latent_dim[2], latent_dim[1]),
            nn.ReLU(),
            nn.Linear(latent_dim[1], latent_dim[0]),
            nn.ReLU(),
            nn.Linear(latent_dim[0], input_dim),
            nn.ReLU()
        )
 
    def encode(self, x):
        fore1 = self.encode_layers(x)
        mean = self.mean(fore1)
        log_var = self.log_var(fore1)
    
        return mean, log_var
 
    def reparameterization(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma
 
    def decode(self, z):
        recon_x = self.decode_layers(z)
        return recon_x
    
    def forward(self, x):
        org_size = x.size()
        batch_size = org_size[0]
        x = x.view((batch_size, -1))
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decode(z).view(org_size)
        return z, recon_x, mean, log_var




class Classifier(nn.Module):
    '''modified from https://blog.csdn.net/HYJ15679729212/article/details/132242694
    '''
    def __init__(self, input_dim: int=50, latent_dim: list = [128, 64], output_dim: int = 10):
        super(Classifier, self).__init__()

        self.layers = []
        self.layers.append(nn.Linear(input_dim, latent_dim[0]))
        self.layers.append(nn.ReLU())

        for i, num in enumerate(latent_dim[0:-1]):
            self.layers.append(nn.Linear(num, latent_dim[i+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(latent_dim[-1], output_dim))

        # self.classifier_layers = nn.Sequential(
        #     nn.Linear(input_dim, latent_dim[0]),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim[0], latent_dim[1]),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim[1], output_dim),
        # )
    
        self.classifier_layers = nn.Sequential(*self.layers)

    def classifier(self, mean):
        output = self.classifier_layers(mean)
        return output
    
    def forward(self, x, model_vae):    # 加入VAE
 
        # mean, log_var = model_vae.encode(x)
        # pred_y = self.classifier(mean)

        # z, recon_x, mean, log_var = model_vae(x)
        # pred_y = self.classifier(recon_x)

        pred_y = self.classifier(x)

        return pred_y




class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=hidden_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels,
                      out_channels=out_channels,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, hidden_channels):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, out_channels, hidden_channels)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    


class ResNet(nn.Module):
    def __init__(self, num_classes=None, doc:bool = False):
        super(ResNet, self).__init__()
        self.output_class = num_classes
        out_channels = 64
        hidden_channels = 1024
        num_residual_layers = 20

        self._conv_1 = nn.Conv1d(in_channels=1,
                                 out_channels=64,
                                 kernel_size=7,
                                 stride=1, padding=1)
        self._bn_1 = nn.BatchNorm1d(64)
        self._conv_1_downsampler = nn.Conv1d(in_channels=64,
                                             out_channels=64,
                                             kernel_size=4,
                                             stride=2, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=7,
                                 stride=1, padding=1)
        self._bn_2 = nn.BatchNorm1d(128)
        self._conv_2_downsampler = nn.Conv1d(in_channels=128,
                                             out_channels=128,
                                             kernel_size=4,
                                             stride=2, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=128,
                                 out_channels=256,
                                 kernel_size=7,
                                 stride=1, padding=1)
        self._bn_3 = nn.BatchNorm1d(256)
        self._conv_3_downsampler = nn.Conv1d(in_channels=256,
                                             out_channels=256,
                                             kernel_size=4,
                                             stride=2, padding=1)
        self._conv_4 = nn.Conv1d(in_channels=256,
                                 out_channels=512,
                                 kernel_size=7,
                                 stride=1, padding=1)
        self._bn_4 = nn.BatchNorm1d(512)
        self._conv_4_downsampler = nn.Conv1d(in_channels=512,
                                             out_channels=512,
                                             kernel_size=4,
                                             stride=2, padding=1)
        self._conv_5 = nn.Conv1d(in_channels=512,
                                 out_channels=512,
                                 kernel_size=7,
                                 stride=1, padding=1)
        self._bn_5 = nn.BatchNorm1d(512)
        self._conv_5_downsampler = nn.Conv1d(in_channels=512,
                                             out_channels=512,
                                             kernel_size=4,
                                             stride=2, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=512,
                                             out_channels=512,
                                             num_residual_layers=num_residual_layers,
                                             hidden_channels=512)
        self._conv_final = nn.Conv1d(in_channels=512,
                                     out_channels=out_channels,
                                     kernel_size=1)

        self.drop0 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(1728, 256) 
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(256, 256) 
        self.drop2 = nn.Dropout(p=0.5)

        self.fc3 = None if num_classes is None else nn.Linear(256, num_classes)


    def forward(self, x):
        x = self._conv_1(x)
        x = self._bn_1(x)
        x = F.relu(x)
        x = self._conv_1_downsampler(x)
        x = F.relu(x)

        x = self._conv_2(x)
        x = self._bn_2(x)
        x = F.relu(x)
        x = self._conv_2_downsampler(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = self._bn_3(x)
        x = F.relu(x)
        x = self._conv_3_downsampler(x)
        x = F.relu(x)

        x = self._conv_4(x)
        x = self._bn_4(x)
        x = F.relu(x)
        x = self._conv_4_downsampler(x)
        x = F.relu(x)

        x = self._conv_5(x)
        x = self._bn_5(x)
        x = F.relu(x)
        x = self._conv_5_downsampler(x)
        x = F.relu(x)

        x = self._residual_stack(x)

        x = self._conv_final(x)
        x = F.relu(x)
        # print(x.shape)

        out = torch.flatten(x, start_dim=1)
        # print(out.shape)
        out = self.drop0(out)

        out = self.fc1(out)
        out = self.drop1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = self.drop2(out)
        out = F.relu(out)

        out = self.fc3(out)

        return out

