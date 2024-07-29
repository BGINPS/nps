# import torch
# from torch import nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from . import model as md
# import numpy as np
# import pandas as pd
# import random
import os
# import seaborn as sns
# from matplotlib.patches import Rectangle
# import matplotlib.pyplot as plt



# DEFAULT_RANDOM_SEED = 2023

# def seedBasic(seed=DEFAULT_RANDOM_SEED):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
    
# # torch random seed
# def seedTorch(seed=DEFAULT_RANDOM_SEED):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
      
# # basic + tensorflow + torch 
# def seed_everything(seed=DEFAULT_RANDOM_SEED):
#     seedBasic(seed)
#     seedTorch(seed)


# def doc_loss(
#     preds: torch.Tensor, 
#     targets: torch.Tensor, 
#     eps: float=1e-5
# ) -> torch.Tensor:
#     """Calculate doc loss

#     Args:
#         preds (torch.Tensor): N * M tensor. N: sample number, M: class number. all elements are in 0-1
#         targets (torch.Tensor): the same shape with preds, all elements are either 0 or 1, indicating class
#         eps (float, optional): eps. Defaults to 1e-10.

#     Returns:
#         torch.Tensor: doc loss
#     """
#     # DOC: Deep Open Classification of Text Documents
#     # Lei Shu, Hu Xu, Bing Liu
#     # p = preds + eps # eps --- 1+eps
#     p = preds
#     min_c = torch.Tensor([eps])
#     max_c = torch.Tensor([1-eps])
#     if p.is_cuda:
#         min_c = min_c.to(torch.device('cuda:0'))
#         max_c = max_c.to(torch.device('cuda:0'))
        
#     p = torch.max(p, min_c)
#     p = torch.min(p, max_c)
#     loss = torch.sum(-torch.log(p)*targets) + torch.sum(-torch.log(1-p)*(1-targets))
#     return loss

# class Trainer():
#     def __init__(self, train_dataloader: DataLoader, test_dataloader, out_class: int=None, eva_test_dataset_each_epoch:bool = True, device='gpu', lr=0.05, epoch=200, doc:bool=False, alpha=3.0, model='LSTM'):
#         self.lr = lr
#         self.epoch = epoch
#         self.train_dataloader = train_dataloader
#         self.test_dataloader = test_dataloader
#         self.output_class = out_class
#         self.eva_test_dataset_each_epoch = eva_test_dataset_each_epoch
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() and device.upper()=='GPU' else 'cpu')
#         self.doc = doc

#         if out_class is None:
#             self.output_class = len(self.train_dataloader.dataset.y[0])
#         if not self.doc:
#             assert self.output_class == len(self.test_dataloader.dataset.y[0])
#         self.alpha = alpha
#         if model.upper() == 'CNN':
#             self.model = md.CNN1D(num_classes=self.output_class, doc=self.doc)
#         elif model.upper() == 'RESNET':
#             self.model = md.ResNet(num_classes=self.output_class, doc=self.doc)
#         elif model.upper() == "LSTM":
#             self.model = md.CnnLstmNet(output_class=self.output_class, doc=self.doc)
#         elif model.upper() == 'CNN_STFT':
#             self.model = md.CNN1D_Stft(num_classes=self.output_class, doc=self.doc)
#         elif model.upper() == 'LSTM_STFT':
#             self.model = md.LSTM_Stft(num_classes=self.output_class, doc=self.doc)
#         else:
#             print('please give the correct model name')
#         self.loss_fun = doc_loss if self.doc else nn.CrossEntropyLoss(reduction='sum')
#         self.doc_thresholds = (torch.ones([self.output_class]) * 0.5).to(self.device)
#         # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#         self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
#         # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, min_lr=self.lr*0.01, 
#         #                                                             verbose=True, patience=3)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, 
#                                                          gamma=0.5, verbose=True)


#         self.model.to(self.device)

#         print(f'total parameter number: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        
    
#     def train(self,):
#         writer = SummaryWriter()
#         for epoch in range(self.epoch):
#             # train loss
#             self.model.train()
#             losses_in_an_epoch, accs_num_in_an_epoch, sample_num_in_an_epoch, all_outputs_for_an_epoch, all_ys_for_an_epoch = [], [], [], [], []
#             for indx, (_, X, y) in enumerate(self.train_dataloader):
#                 # X = X.unsqueeze(1)
#                 X, y = X.to(self.device), y.to(self.device)
#                 outputs = self.model(X)
#                 # print(outputs.shape)
#                 all_outputs_for_an_epoch.append(outputs)
#                 all_ys_for_an_epoch.append(y)
#                 loss = self.loss_fun(outputs, y)
                
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 losses_in_an_epoch.append(loss.item())
#                 accs_num_in_an_epoch.append(torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)).item())
#                 sample_num_in_an_epoch.append(len(X))
#             if epoch % 10 ==0:
#                 all_outputs_for_an_epoch = torch.cat(all_outputs_for_an_epoch, dim=0)
#                 all_ys_for_an_epoch = torch.cat(all_ys_for_an_epoch, dim=0)
#                 if self.doc:
#                     self.update_doc_thresholds(all_outputs_for_an_epoch, all_ys_for_an_epoch)
#                     print(self.doc_thresholds)
#             train_acc = np.sum(accs_num_in_an_epoch) / np.sum(sample_num_in_an_epoch)
#             print(f'epoch: {epoch + 1:5d}, training loss: {np.sum(losses_in_an_epoch) / np.sum(sample_num_in_an_epoch):.8f} \
#                   training acc: {train_acc:.4f}', end="\t")
#             # writer.add_scalar("Acc/train", train_acc, epoch)
    

#             if not self.eva_test_dataset_each_epoch:
#                 print('\n', end='')
#                 continue
#             # test loss
#             self.model.eval()
#             test_losses_in_an_epoch, test_accs_num_in_an_epoch, test_sample_num_in_an_epoch = [], [], []
#             with torch.no_grad():
#                 for indx, (_, X, y) in enumerate(self.test_dataloader):
#                     # X = X.unsqueeze(1)
#                     X, y = X.to(self.device), y.to(self.device)
#                     outputs = self.model(X)
#                     if self.doc:
#                         loss = self.loss_fun(outputs[y[:,-1]==0], y[y[:,-1]==0,0:-1])
#                     else:
#                         loss = self.loss_fun(outputs, y)
#                     test_losses_in_an_epoch.append(loss.item())
#                     test_accs_num_in_an_epoch.append(torch.sum(self.output_to_class(outputs) == torch.argmax(y, dim=1)).item())
#                     test_sample_num_in_an_epoch.append(len(X))
#             test_acc = np.sum(test_accs_num_in_an_epoch) / np.sum(test_sample_num_in_an_epoch)
#             print(f' testing loss: {np.sum(test_losses_in_an_epoch) / np.sum(test_sample_num_in_an_epoch):.8f} \
#                   testing acc: {test_acc:.4f}')
#             # self.scheduler.step(test_acc)
#             self.scheduler.step()
#             writer.add_scalars("Acc", {'train_acc': train_acc, 'test_acc': test_acc}, epoch)
#         writer.flush()
#         writer.close()


#     def update_doc_thresholds(self, all_outputs_for_an_epoch, all_ys_for_an_epoch):
#         for c in range(all_ys_for_an_epoch.shape[1]):
#             probs = all_outputs_for_an_epoch[all_ys_for_an_epoch[:,c]==1,c]
#             pseudo_probs = torch.Tensor([1-i+1 for i in probs]).to(self.device)
#             all_probs = torch.cat([probs, pseudo_probs])
#             self.doc_thresholds[c] = max(0.5, 1 - all_probs.std() * self.alpha)


#     def output_to_class(self, outputs):
#         class_labels = torch.argmax(outputs, dim=1)
#         if self.doc:
#             lower_than_cut = torch.zeros_like(outputs)
#             lower_than_cut[outputs < self.doc_thresholds] = 1
#             class_labels[torch.sum(lower_than_cut, dim=1) == self.output_class] = self.output_class
#         return class_labels


#     def predict_test_dataset(self,):
#         return self.predict_dataloader(self.test_dataloader)
    
#     def predict_test_dataset_with_label(self, classes):
#         read_ids, preds, trues = self.predict_dataloader(self.test_dataloader)
#         df = pd.DataFrame({'read_id': read_ids, 'pred': preds, 'true': trues})
#         c = np.array(classes)
#         df['pred'] = c[df['pred']]
#         df['true'] = c[df['true']]
#         df = df.set_index('read_id')
#         return df

#     def predict_train_dataset(self,):
#         return self.predict_dataloader(self.train_dataloader)

#     def predict_dataloader(self, dataloader):
#         dataloader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False, drop_last=False)
#         self.model.eval()
#         all_outputs, ys, all_read_ids = [], [], []
#         with torch.no_grad():
#             for indx, (read_ids, X, y) in enumerate(dataloader):
#                 # X = X.unsqueeze(1)
#                 X, y = X.to(self.device), y.to(self.device)
#                 all_outputs.append(self.model(X))
#                 ys.append(y)
#                 all_read_ids.append(read_ids)
#         all_preds = self.output_to_class(torch.cat(all_outputs, dim=0)).cpu().numpy()
#         all_ys = torch.argmax(torch.cat(ys, dim=0), dim=1).cpu().numpy()
#         all_read_ids = np.concatenate(all_read_ids)
#         return all_read_ids, all_preds, all_ys
    

#     def get_cm(self, true, pred, class_labels, fill_empty_class=True):
#         cm_df = pd.DataFrame({'pred': pred, 'true': true})
#         cm_df['count'] = 1
#         cm_df = cm_df.groupby(["pred", 'true']).sum().reset_index()
#         if fill_empty_class:
#             for i in np.setdiff1d(np.unique(true), np.unique(pred)):
#                 cm_df.loc[len(cm_df)] = [i,0,0]
#         cm_df = cm_df.pivot(index='true', columns='pred', values='count').fillna(0)
#         cm_df.columns = class_labels
#         cm_df.index = class_labels
#         cm_df.columns.name = 'pred'
#         cm_df.index.name = 'true'
#         return cm_df


#     def plot_cm_for_test_dataset(
#         self,
#         class_labels: list,
#         nor_to_percent_for_each_pred: bool = True,
#         mark_diagonal_line: bool = True,
#         lw_of_rectangle: int = 1,
#         annot: bool = True,
#         annot_size=8,
#         figsize=(10,10),
#         save_cm_to_file=None,
#     ):
        
#         read_ids, pred, true = self.predict_test_dataset()
#         return self.cal_cm_and_plot(class_labels, nor_to_percent_for_each_pred, mark_diagonal_line, lw_of_rectangle, annot, annot_size, pred, true, 
#                                     save_cm_to_file=save_cm_to_file, figsize=figsize)
    

#     def plot_cm_for_train_dataset(
#         self,
#         class_labels: list,
#         nor_to_percent_for_each_pred: bool = True,
#         mark_diagonal_line: bool = True,
#         lw_of_rectangle: int = 1,
#         annot: bool = True,
#         annot_size=8,
#         figsize=(10,10)
#     ):
#         if self.doc:
#             self.doc = False
#             _, pred, true = self.predict_train_dataset()
#             self.doc = True
#         else:
#             _, pred, true = self.predict_train_dataset()
#         ax = self.cal_cm_and_plot(class_labels, nor_to_percent_for_each_pred, mark_diagonal_line, lw_of_rectangle, annot, annot_size, pred, true, figsize=figsize)
#         return ax

#     def cal_cm_and_plot(self, class_labels, nor_to_percent_for_each_pred, mark_diagonal_line, lw_of_rectangle, annot, annot_size, pred, true, save_cm_to_file=None, figsize=(10,10)):
#         cm_df = self.get_cm(true, pred, class_labels)
#         print(cm_df)
#         if save_cm_to_file:
#             cm_df.to_csv(save_cm_to_file)
#         ax = self.plot_cm(cm_df=cm_df, 
#                           nor_to_percent_for_each_pred=nor_to_percent_for_each_pred,
#                           figsize=figsize, 
#                           annot=annot,
#                           annot_size=annot_size,
#                           lw_of_rectangle=lw_of_rectangle,
#                           mark_diagonal_line=mark_diagonal_line)
#         return ax
                

#     def read_csv_of_cm_and_plot(
#         self, 
#         csv_file_of_cm: str,
#         nor_to_percent_for_each_pred: bool = True, 
#         figsize: tuple = (10,10), 
#         annot: bool = True,
#         annot_size: float = 8.0,
#         lw_of_rectangle: float = 1.0,
#         mark_diagonal_line: bool = True
#     ):
#         cm_df = pd.read_csv(csv_file_of_cm, index_col=0)
#         cm_df.columns.name = 'pred'
#         cm_df.index.name = 'true'
#         ax = self.plot_cm(cm_df=cm_df, 
#                      nor_to_percent_for_each_pred=nor_to_percent_for_each_pred,
#                      figsize=figsize,
#                      annot=annot,
#                      annot_size=annot_size,
#                      lw_of_rectangle=lw_of_rectangle,
#                      mark_diagonal_line=mark_diagonal_line)
#         return ax


#     def plot_cm(
#         self,
#         cm_df,
#         nor_to_percent_for_each_pred: bool = True, 
#         figsize: tuple = (10,10), 
#         annot: bool = True,
#         annot_size: float = 8.0,
#         lw_of_rectangle: float = 1.0,
#         mark_diagonal_line: bool = True
#     ):
#         if nor_to_percent_for_each_pred:
#             cm_df = cm_df.div(cm_df.sum(axis=1), axis='rows')
#         fig, ax = plt.subplots(figsize=figsize)
#         sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_size}, cmap="YlOrRd", fmt=".1%", ax=ax, vmin=0, vmax=1)

#         if mark_diagonal_line:
#             for i in range(len(cm_df)):
#                 ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=lw_of_rectangle, clip_on=False))
#         return ax


                




            
# class VAETrainer():
#     def __init__(self, train_dataloader, test_dataloader, eva_test_dataset_each_epoch:bool = True, device='gpu', lr=0.05, epoch=200, doc:bool=False, alpha=3.0,):
#         self.lr = lr
#         self.epoch = epoch
#         self.train_dataloader = train_dataloader
#         self.test_dataloader = test_dataloader
#         self.eva_test_dataset_each_epoch = eva_test_dataset_each_epoch
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() and device.upper()=='GPU' else 'cpu')

#         self.output_class = len(self.train_dataloader.dataset.y[0])
        
#         self.vae_model = md.VAE(input_dim=1000, latent_dim=[256, 128, 64, 50])
#         self.cls_model = md.Classifier(input_dim=1000, latent_dim=[1024, 1024, 1024, 1024, 1024], output_dim=self.output_class)

#         self.recon_loss_fun = nn.MSELoss(reduction='mean')
#         self.kl_loss_fun = lambda mean, log_var: -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
#         self.classification_loss_fun = nn.CrossEntropyLoss(reduction='sum')

#         self.optimizer = optim.Adam(self.vae_model.parameters(), lr=self.lr)
#         self.optimizer_cls = optim.Adam(self.cls_model.parameters(), lr=self.lr)

#         self.vae_model.to(self.device)
#         self.cls_model.to(self.device)

#         print(f'total parameter number of vae model: {sum(p.numel() for p in self.vae_model.parameters() if p.requires_grad)}')
#         print(f'total parameter number of cls model: {sum(p.numel() for p in self.cls_model.parameters() if p.requires_grad)}')
        
    
#     def train_vae(self,):
#         for epoch in range(self.epoch):
#             # train loss
#             self.vae_model.train()
#             losses_in_an_epoch, sample_num_in_an_epoch = [], []
#             for indx, (_, X, y) in enumerate(self.train_dataloader):
#                 X, y = X.to(self.device), y.to(self.device)
#                 z, recon_x, mean, log_var = self.vae_model(X)
#                 # print(indx)
#                 # print(z)
#                 # print(X)
#                 # print(self.recon_loss_fun(recon_x, X))
#                 # print(self.kl_loss_fun(mean, log_var))
#                 loss = self.recon_loss_fun(recon_x, X) + self.kl_loss_fun(mean, log_var)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 losses_in_an_epoch.append(loss.item())
#                 sample_num_in_an_epoch.append(len(X))
            
#             print(f'epoch: {epoch + 1:5d}, training loss: {np.mean(losses_in_an_epoch):.8f}')
        
#     def train_cls(self, ):
#         self.vae_model.eval()
#         for epoch in range(self.epoch):
#             self.cls_model.train()
#             losses_in_an_epoch, accs_num_in_an_epoch, sample_num_in_an_epoch, all_outputs_for_an_epoch, all_ys_for_an_epoch = [], [], [], [], []
#             for indx, (_, X, y) in enumerate(self.train_dataloader):
#                 X, y = X.to(self.device), y.to(self.device)
#                 outputs = self.cls_model(X, self.vae_model)
#                 outputs.squeeze_(1)
#                 all_outputs_for_an_epoch.append(outputs)
#                 all_ys_for_an_epoch.append(y)
#                 loss = self.classification_loss_fun(outputs, y)
#                 # print(outputs[0,:])
#                 # print(y[0,:])
#                 # self.optimizer.zero_grad()
#                 self.optimizer_cls.zero_grad()
#                 loss.backward()
#                 # self.optimizer.step()
#                 self.optimizer_cls.step()
#                 losses_in_an_epoch.append(loss.item())
#                 accs_num_in_an_epoch.append(torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)).item())
#                 sample_num_in_an_epoch.append(len(X))

#             print(f'epoch: {epoch + 1:5d}, training loss: {np.sum(losses_in_an_epoch) / np.sum(sample_num_in_an_epoch):.8f} \
#                   training acc: {np.sum(accs_num_in_an_epoch) / np.sum(sample_num_in_an_epoch):.4f}', end="\n")

    