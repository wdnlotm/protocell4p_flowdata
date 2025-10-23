import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from load_data_cytof_ver013 import *
from model_ver013 import *
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pickle
from tqdm import trange

class Logger:
  
    def __init__(self, path, log=True):
        self.path = path
        self.log = log

    def __call__(self, content, **kwargs):
        # print(content, **kwargs)
        if self.log:
            with open(self.path, 'a') as f:
                print(content, file=f, **kwargs)

class Config:
    
    def __init__(self, data="lupus", data_loc = "../data", model="ProtoCell", split_ratio = [0.5, 0.25, 0.25], \
                 lr = 1e-4, max_epoch = 20, batch_size = 4, test_step = 1, h_dim = 128, z_dim = 32, n_layers = 2, \
                 n_proto=8, device="cpu", seed=0, exp_str=None, task=None, subsample=False, \
                 eval=False, load_ct=True, keep_sparse=True, d_min=1, log_file_name="log1.txt", \
                 lambda_1=1, lambda_2=1, lambda_3=1, lambda_4=1, lambda_5=1, \
                 lambda_6=1, pretrained=False, max_epoch_pretrain=0, lr_pretrain=1e-2):

        assert len(split_ratio) == 3 and sum(split_ratio) == 1.0

        self.data_loc = data_loc
        self.log_file_name = log_file_name
        self.log_dir4model_log = os.path.join("../log", data, exp_str)

        self.result_df = pd.DataFrame()


        self.seed = seed
        self.keep_sparse = keep_sparse

        # load the target data
        if data.lower() == "lupus":
            assert task is not None
            dataset = load_lupus(task=task, load_ct=load_ct, keep_sparse=self.keep_sparse)
            self.collate_fn = self.my_collate
        elif data.lower() == "cardio":
            dataset = load_cardio(load_ct=load_ct, keep_sparse=self.keep_sparse)
        elif data.lower() == "covid":
            dataset = load_covid(load_ct=load_ct, keep_sparse=self.keep_sparse)
        elif data.lower() == "gausses":
            dataset = load_gausses(load_ct=load_ct, keep_sparse=self.keep_sparse)
        elif data.lower() == "cytof":
            dataset = load_cytof(data_dir=self.data_loc, load_ct=load_ct, keep_sparse=self.keep_sparse, data_location = self.data_loc)
            # dataset = load_cytof(load_ct=load_ct, keep_sparse=self.keep_sparse)
        else:
            raise ValueError("Data [:s] not supported".format(data))

        self.collate_fn = self.my_collate

        # split the dataset according to train_val_test_split
        self.train_set, self.test_set = train_test_split(dataset, test_size=split_ratio[2], random_state=self.seed)
        self.train_set, self.val_set = train_test_split(self.train_set, test_size=split_ratio[1] / (1-split_ratio[2]), random_state=self.seed)

        patient_list = patid_identify(csv_file_name = "patid_count.csv", train_data =self.train_set, val_data =self.val_set, test_data = self.test_set)
        #print(patient_list)
        for ii in patient_list:
            self.result_df[ii]=()

        result_df_col_list = ["loss_train", "accuracy_train", "roc_auc_train", "f1_train", "clf_loss_train", "ct_loss_train", "ct_acc_train", "loss_val", "accuracy_val", "roc_auc_val", "f1_val", "clf_loss_val", "ct_loss_val", "ct_acc_val"]

        for ii in result_df_col_list :
            self.result_df[ii]=()
        
        print(self.result_df.columns)
        
        self.device = device
        self.lr = lr
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.test_step = test_step
        # self.save_step = save_step
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_classes = max(dataset.y) + 1
        self.n_proto = n_proto
        if dataset.ct is None:
            self.n_ct = None
        else:
            self.n_ct = len(dataset.ct_id)
        self.ct_id = dataset.ct_id
        self.class_id = dataset.class_id
        self.pretrained = pretrained
        self.max_epoch_pretrain = max_epoch_pretrain
        self.lr_pretrain = lr_pretrain

        self.model_type = model
        self.lambdas = {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "lambda_3": lambda_3,
            "lambda_4": lambda_4,
            "lambda_5": 0 if self.n_ct is None else lambda_5,
            "lambda_6": 0 if self.n_ct is None else lambda_6
        }
        
        if self.model_type == "ProtoCell":
            self.model = ProtoCell(dataset.X[0].shape[1], self.h_dim, self.z_dim, self.n_layers, \
                self.n_proto, self.n_classes, self.lambdas, self.n_ct, self.device, d_min=d_min, \
                                   log_dir = self.log_dir4model_log, log_file = self.log_file_name)  
        elif self.model_type == "BaseModel":
            self.model = BaseModel(dataset.X[0].shape[1], self.h_dim, self.z_dim, self.n_layers, \
                self.n_proto, self.n_classes, self.lambdas, self.n_ct, self.device, d_min=d_min) 
        else:
            raise ValueError("Model [:s] not supported".format(self.model_type))

        self.model.to(self.device)

        assert exp_str is not None
     
        if data.lower() == "lupus":
            self.checkpoint_dir = os.path.join("../checkpoint", data, task, exp_str)
            self.log_dir = os.path.join("../log", data, task, exp_str)
        else:
            self.checkpoint_dir = os.path.join("../checkpoint", data, exp_str)
            self.log_dir = os.path.join("../log", data, exp_str)

        if eval:
            self.logger = Logger(os.path.join(self.log_dir, self.log_file_name), log=False)  #"log.txt"
        else:
            self.logger = Logger(os.path.join(self.log_dir, self.log_file_name))  #"log.txt"
            self.tf_writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

        if subsample:
            c_counts = []
            for ins in self.train_set:
                c_counts.append(ins[0].shape[0])
            W = sum(c_counts) // (5 * len(c_counts))

            train_set = []
            for ins in self.train_set:
                n = ins[0].shape[0]
                if n < W * 2:
                    train_set.append(ins)
                    continue
                elif n < W * 4:
                    D = round(2 * n / W) # number of times for subsampling
                else:
                    D = 8
                idx = np.arange(n)
                for _ in range(D):
                    np.random.shuffle(idx)
                    if self.n_ct is None:
                        train_set.append([ins[0][idx[:W]], ins[1]])
                    else:
                        train_set.append([ins[0][idx[:W]], ins[1], [ins[2][i] for i in idx[:W]]])
            self.logger("Training Data Augmented: {:d} --> {:d}".format(len(self.train_set), len(train_set)))
            self.train_set = train_set
            self.train_batch_size = 8 * self.batch_size # 64 for the 625 setting
        else:
            self.train_batch_size = self.batch_size

        self.logger("*" * 40)
        self.logger("Learning rate: {:f}".format(self.lr))
        self.logger("Max epoch: {:d}".format(self.max_epoch))
        self.logger("Batch size: {:d}".format(self.batch_size))
        self.logger("Test step: {:d}".format(self.test_step))
        # self.logger("Save step: {:d}".format(self.save_step))
        self.logger("Number of hidden layers: {:d}".format(self.n_layers))
        self.logger("Number of prototypes: {:d}".format(self.n_proto))
        self.logger("Lambdas: {}".format(self.lambdas))
        self.logger("D_min: {}".format(d_min))
        self.logger("Device: {:s}".format(self.device))
        self.logger("Seed: {}".format(self.seed))
        # self.logger("h_dim: {:d}".format(self.h_dim))
        self.logger("*" * 40)
        self.logger("")

    def train(self):
                
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.pretrained:
            print('Executing pretrain() in config.py')
            self.pretrain()
            # load pretrained model
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "pretrain.pt"), map_location=self.device))
            # fix the parameters of the encoder (everything except the importance scorer)
            for name, param in self.model.named_parameters():
                # if not name.startswith("proto") and not name.startswith("imp") and not name.startswith("ct_clf2") and not name.startswith("clf"):
                if not name.startswith("imp") and not name.startswith("ct_clf2") and not name.startswith("clf"):
                    param.requires_grad = False

        #train_loader shuffle is changed to False, Myles.
        train_loader = DataLoader(self.train_set, batch_size = self.train_batch_size, shuffle = False, collate_fn=self.collate_fn)
        val_loader = DataLoader(self.val_set, batch_size = self.batch_size, shuffle = False, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.test_set, batch_size = self.batch_size, shuffle = False, collate_fn=self.collate_fn)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_epoch = 0
        best_metric = 0


        self.model.train()
        self.logger("Training Starts\n")

        pbar = trange(self.max_epoch, unit=" epoch") #myles
                          #range(self.max_epoch)
        for epoch in pbar: # range(self.max_epoch):
            #print(epoch)  #myles
            pbar.set_description("training ")
            epoch_copy = epoch
                                    
            self.logger("[Epoch {:d}]".format(epoch))
            train_loss = 0
            train_clf_loss = 0
            train_ct_loss = 0
            
            start_time = time.time()
            y_truth = []
            y_logits = []
            if self.n_ct is not None:
                ct_truth = []
                ct_logits = []

            for bat in train_loader:
                x = bat[0]
                y = bat[1]
                optim.zero_grad()
                if self.n_ct is not None:
                    #print('Executing loss, logits, ct_logit = self.model')             #myles added this below       
                    loss, logits, ct_logit, clf_loss_copy, ct_loss_copy= self.model(*bat, sparse=self.keep_sparse, epoch_num = epoch_copy)    
                else:
                    loss, logits = self.model(*bat, sparse=self.keep_sparse)
                
                if not self.pretrained:
                    #print('Executing loss plus = self.model.pretrain')
                    if self.n_ct is not None:
                        loss += self.model.pretrain(*bat, sparse=self.keep_sparse, epoch_num = epoch_copy)[0]
                    else:
                        loss += self.model.pretrain(*bat, sparse=self.keep_sparse)                    
                loss.backward()
                optim.step()
                train_loss += loss.item() * len(x)
                train_clf_loss = clf_loss_copy
                train_ct_loss = ct_loss_copy
                y_truth.append(y)
                y_logits.append(torch.softmax(logits, dim=1))
                if self.n_ct is not None:
                    ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                    ct_logits.append(torch.softmax(ct_logit, dim=1))


            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)
            y_pred = y_logits.argmax(dim=1)

            #print(y_pred.cpu())#myles
            tr_size = len(self.train_set)
            self.result_df.loc[len(self.result_df)]=0.0  #initialize a new row
            self.result_df.iloc[len(self.result_df)-1,0:tr_size]=y_pred.cpu().numpy()[0:tr_size] #update training prediction
            #Myles print
            #print(self.result_df)
            # if epoch % 100 ==1:
            #     print(y_pred.cpu().numpy()[0:tr_size])
            #     print(y_truth)
            

            train_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
            if y_logits.shape[1] == 2:
                train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
            else:
                train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
            # train_f1 = f1_score(y_truth.cpu(), y_pred.cpu())
            train_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")

            if self.n_ct is not None:
                #print('hi from config 251')   # myles
                ct_truth = torch.cat(ct_truth)
                ct_logits = torch.cat(ct_logits)
                ct_pred = ct_logits.argmax(dim=1)
                ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                self.logger("Time: {:.1f}s | Avg. Training Loss: {:.2f} | Avg. Training Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}".format(time.time() - start_time, train_loss / len(self.train_set), train_acc, train_auc, train_f1, ct_acc))
            else:
                self.logger("Time: {:.1f}s | Avg. Training Loss: {:.2f} | Avg. Training Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}".format(time.time() - start_time, train_loss / len(self.train_set), train_acc, train_auc, train_f1))

            self.tf_writer.add_scalar("loss/train", train_loss / len(self.train_set), epoch)
            self.tf_writer.add_scalar("accuracy/train", train_acc, epoch)
            self.tf_writer.add_scalar("roc_auc/train", train_auc, epoch)
            self.tf_writer.add_scalar("f1/train", train_f1, epoch)
            self.tf_writer.add_scalar("clf_loss/train", train_clf_loss, epoch)
            self.tf_writer.add_scalar("ct_loss/train", train_ct_loss, epoch)
            if self.n_ct is not None:
                self.tf_writer.add_scalar("ct_acc/train", ct_acc, epoch)

            #Result df update
            #print(train_clf_loss)
            #print(self.result_df.columns)
            self.result_df.loc[len(self.result_df)-1,'loss_train'] = train_loss / len(self.train_set)
            self.result_df.loc[len(self.result_df)-1,'accuracy_train'] = train_acc
            self.result_df.loc[len(self.result_df)-1,'roc_auc_train'] = train_auc
            self.result_df.loc[len(self.result_df)-1,'f1_train'] = train_f1
            self.result_df.loc[len(self.result_df)-1,'clf_loss_train'] = train_clf_loss.detach().cpu().numpy()
            self.result_df.loc[len(self.result_df)-1,'ct_loss_train'] = train_ct_loss.detach().cpu().numpy()
            self.result_df.loc[len(self.result_df)-1,'ct_acc_train'] = ct_acc
            
            ##Myles added
            if epoch in [1, 2, 3, 4, 7, 12, 20, 30, 50, 90, 120, 200, 300, 500, 800, 1200, 1600, 2000, 2500, 3000]:
                model_file_name = "model_at_"+str(epoch)+".pt"
                torch.save(self.model.state_dict(), \
                    os.path.join(self.checkpoint_dir, model_file_name))
            ## Myles added end
            
            ##Myles added
            if epoch == (self.max_epoch-1):
                torch.save(self.model.state_dict(), \
                    os.path.join(self.checkpoint_dir, "last_model.pt"))
            ## Myles added end
            
            if (epoch + 1) % self.test_step == 0:
                self.model.eval()
                self.logger("[Evaluation on Validation Set]")
                start_time = time.time()
                val_loss = 0
                vel_clf_loss = 0
                val_ct_loss = 0
                y_truth = []
                y_logits = []
                if self.n_ct is not None:
                    ct_truth = []
                    ct_logits = []

                with torch.no_grad():

                    for bat in val_loader:
                        x = bat[0]
                        y = bat[1]
                        #print('Executing loss, logits, ct_logit = self.model')
                        if self.n_ct is not None:                                              #myles added this below
                            loss, logits, ct_logit, clf_loss_copy, ct_loss_copy= self.model(*bat, sparse=self.keep_sparse, epoch_num = epoch_copy+10000)
                        else:
                            loss, logits = self.model(*bat, sparse=self.keep_sparse)
                        val_loss += loss.item() * len(x)
                        val_clf_loss = clf_loss_copy
                        val_ct_loss = ct_loss_copy
                        y_truth.append(y)
                        y_logits.append(torch.softmax(logits, dim=1))
                        if self.n_ct is not None:
                            ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                            ct_logits.append(torch.softmax(ct_logit, dim=1))

                    y_truth = torch.cat(y_truth)
                    y_logits = torch.cat(y_logits)
                    y_pred = y_logits.argmax(dim=1)

                    val_size = len(self.val_set)
                    #self.result_df.loc[len(self.result_df)]=0 #No need to initialize
                    self.result_df.iloc[len(self.result_df)-1,tr_size:(val_size+tr_size)]=y_pred.cpu().numpy()[0:val_size]
                    #print(self.result_df)

                    val_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
                    if y_logits.shape[1] == 2:
                        val_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
                    else:
                        val_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
                    # val_f1 = f1_score(y_truth.cpu(), y_pred.cpu())
                    val_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")                     

                if self.n_ct is not None:
                    ct_truth = torch.cat(ct_truth)
                    ct_logits = torch.cat(ct_logits)
                    ct_pred = ct_logits.argmax(dim=1)
                    ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                    # curr_metric = val_f1 + ct_acc / 10
                    curr_metric = val_auc + ct_acc / 10
                    self.logger("Time: {:.1f}s | Avg. Validation Loss: {:.2f} | Avg. Validation Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}".format(time.time() - start_time, val_loss / len(self.val_set), val_acc, val_auc, val_f1, ct_acc))
                else:
                    curr_metric = val_auc
                    self.logger("Time: {:.1f}s | Avg. Validation Loss: {:.2f} | Avg. Validation Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}".format(time.time() - start_time, val_loss / len(self.val_set), val_acc, val_auc, val_f1))
                self.tf_writer.add_scalar("loss/val", val_loss / len(self.val_set), epoch)
                self.tf_writer.add_scalar("accuracy/val", val_acc, epoch)
                self.tf_writer.add_scalar("roc_auc/val", val_auc, epoch)
                self.tf_writer.add_scalar("f1/val", val_f1, epoch)
                self.tf_writer.add_scalar("clf_loss/val", val_clf_loss, epoch)
                self.tf_writer.add_scalar("ct_loss/val", val_ct_loss, epoch)
                if self.n_ct is not None:
                    self.tf_writer.add_scalar("ct_acc/val", ct_acc, epoch)

                #Result df update
                self.result_df.loc[len(self.result_df)-1,'loss_val'] = val_loss / len(self.val_set)
                self.result_df.loc[len(self.result_df)-1,'accuracy_val'] = val_acc
                self.result_df.loc[len(self.result_df)-1,'roc_auc_val'] = val_auc
                self.result_df.loc[len(self.result_df)-1,'f1_val'] = val_f1
                self.result_df.loc[len(self.result_df)-1,'clf_loss_val'] = val_clf_loss.detach().cpu().numpy()
                self.result_df.loc[len(self.result_df)-1,'ct_loss_val'] = val_ct_loss.detach().cpu().numpy()
                self.result_df.loc[len(self.result_df)-1,'ct_acc_val'] = ct_acc

                
                ##Myles added
                if (self.max_epoch - epoch < self.test_step+3):
                    model_file_name = "last_model_at_"+str(epoch)+".pt"
                    torch.save(self.model.state_dict(), \
                        os.path.join(self.checkpoint_dir, model_file_name))
                ## Myles added end
                
                if curr_metric > best_metric:
                    torch.save(self.model.state_dict(), \
                        os.path.join(self.checkpoint_dir, "best_model.pt"))

                    best_metric = curr_metric
                    best_epoch = epoch
                    self.logger("Model Saved!")
                        
                self.model.train()

            self.logger("")

        
        self.logger("Best epoch: {:d}".format(best_epoch))
        self.logger("Training Ends\n")

        #Testing the test data with the best model
        
        #change this back - Myles
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "best_model.pt"), map_location=self.device))
        #self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "best_model_epoch3500_123.pt"), map_location=self.device))
        
        self.model.eval()
        self.logger("[Evaluation on Test Set with the best model]")
        start_time = time.time()
        test_loss = 0
        y_truth = []
        y_logits = []
        if self.n_ct is not None:
            ct_truth = []
            ct_logits = []

        with torch.no_grad():

            for bat in test_loader:
                x = bat[0]
                y = bat[1]
                if self.n_ct is not None:
                    loss, logits, ct_logit, clf_loss_copy, ct_loss_copy = self.model(*bat, sparse=self.keep_sparse)
                else:
                    loss, logits = self.model(*bat, sparse=self.keep_sparse)
                test_loss += loss.item() * len(x)
                y_truth.append(y)
                y_logits.append(torch.softmax(logits, dim=1))
                if self.n_ct is not None:
                    ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                    ct_logits.append(torch.softmax(ct_logit, dim=1))

            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)
            y_pred = y_logits.argmax(dim=1)
            print(y_pred)


            print("config 409 y_truth ")
            print(y_truth.cpu())
            print("config 409 y_pred with the best model at " + str(best_epoch))
            print(y_pred.cpu())
            test_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
            if y_logits.shape[1] == 2:
                test_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
            else:
                test_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
            # test_f1 = f1_score(y_truth.cpu(), y_pred.cpu())
            test_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")

        if self.n_ct is not None:
            ct_truth = torch.cat(ct_truth)
            ct_logits = torch.cat(ct_logits)
            ct_pred = ct_logits.argmax(dim=1)
            ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
            self.logger("Time: {:.1f}s | Avg. Test Loss: {:.2f} | Avg. Test Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}\n".format(time.time() - start_time, test_loss / len(self.test_set), test_acc, test_auc, test_f1, ct_acc))
        else:
            self.logger("Time: {:.1f}s | Avg. Test Loss: {:.2f} | Avg. Test Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}\n".format(time.time() - start_time, test_loss / len(self.test_set), test_acc, test_auc, test_f1))
            
        
        test_size = len(self.test_set)
        #self.result_df.loc[len(self.result_df)]=0 #No need to initialize
        self.result_df.iloc[len(self.result_df)-1,(tr_size+val_size):(test_size+val_size+tr_size)] = \
            y_pred.cpu().numpy()
                            
        self.result_df.to_csv("result_"+self.log_file_name.replace(".txt",".csv").replace("log_",""))



    
        #Testing the test data with the best model
        #change this back - Myles
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "last_model.pt"), map_location=self.device))
        #self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "best_model_epoch3500_123.pt"), map_location=self.device))
        
        self.model.eval()
        self.logger("[Evaluation on Test Set with the last model]")
        start_time = time.time()
        test_loss = 0
        y_truth = []
        y_logits = []
        if self.n_ct is not None:
            ct_truth = []
            ct_logits = []

        with torch.no_grad():

            for bat in test_loader:
                x = bat[0]
                y = bat[1]
                if self.n_ct is not None:
                    loss, logits, ct_logit, clf_loss_copy, ct_loss_copy = self.model(*bat, sparse=self.keep_sparse)
                else:
                    loss, logits = self.model(*bat, sparse=self.keep_sparse)
                test_loss += loss.item() * len(x)
                y_truth.append(y)
                y_logits.append(torch.softmax(logits, dim=1))
                if self.n_ct is not None:
                    ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                    ct_logits.append(torch.softmax(ct_logit, dim=1))

            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)
            y_pred = y_logits.argmax(dim=1)

            print("config 409 y_truth ")
            print(y_truth.cpu())
            print("config 409 y_pred ")
            print(y_pred.cpu())
            test_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
            if y_logits.shape[1] == 2:
                test_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
            else:
                test_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
            # test_f1 = f1_score(y_truth.cpu(), y_pred.cpu())
            test_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")

        if self.n_ct is not None:
            ct_truth = torch.cat(ct_truth)
            ct_logits = torch.cat(ct_logits)
            ct_pred = ct_logits.argmax(dim=1)
            ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
            self.logger("Time: {:.1f}s | Avg. Test Loss: {:.2f} | Avg. Test Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}\n".format(time.time() - start_time, test_loss / len(self.test_set), test_acc, test_auc, test_f1, ct_acc))
        else:
            self.logger("Time: {:.1f}s | Avg. Test Loss: {:.2f} | Avg. Test Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}\n".format(time.time() - start_time, test_loss / len(self.test_set), test_acc, test_auc, test_f1))
    
    
    def pretrain(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        train_loader = DataLoader(self.train_set, batch_size = self.train_batch_size, shuffle = True, collate_fn=self.collate_fn)
        val_loader = DataLoader(self.val_set, batch_size = self.batch_size, shuffle = False, collate_fn=self.collate_fn)

        optim_pretrain = torch.optim.Adam(self.model.parameters(), lr=self.lr_pretrain)

        best_epoch = 0
        best_metric = None

        self.model.train()
        self.logger("Pre-training Starts\n")
        
        for epoch in range(self.max_epoch_pretrain):
            self.logger("[Epoch {:d}]".format(epoch))
            train_loss = 0
            start_time = time.time()
            if self.n_ct is not None:
                ct_truth = []
                ct_logits = []

            for bat in train_loader:
                x = bat[0]
                optim_pretrain.zero_grad()
                if self.n_ct is not None:
                    loss, ct_logit = self.model.pretrain(*bat, sparse=self.keep_sparse)
                else:
                    loss = self.model.pretrain(*bat, sparse=self.keep_sparse)
                loss.backward()
                optim_pretrain.step()
                train_loss += loss.item() * len(x)
                if self.n_ct is not None:
                    ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                    ct_logits.append(torch.softmax(ct_logit, dim=1))

            if self.n_ct is not None:
                ct_truth = torch.cat(ct_truth)
                ct_logits = torch.cat(ct_logits)
                ct_pred = ct_logits.argmax(dim=1)
                ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                self.logger("Time: {:.1f}s | Avg. Training Loss: {:.2f} | CT Acc: {:.2f}".format(time.time() - start_time, train_loss / len(self.train_set), ct_acc))
            else:
                self.logger("Time: {:.1f}s | Avg. Training Loss: {:.2f}".format(time.time() - start_time, train_loss / len(self.train_set)))

            if (epoch + 1) % self.test_step == 0:
                self.model.eval()
                self.logger("[Evaluation on Validation Set]")
                start_time = time.time()
                val_loss = 0
                if self.n_ct is not None:
                    ct_truth = []
                    ct_logits = []

                with torch.no_grad():

                    for bat in val_loader:
                        x = bat[0]
                        if self.n_ct is not None:
                            loss, ct_logit = self.model.pretrain(*bat, sparse=self.keep_sparse)    
                        else:
                            loss = self.model.pretrain(*bat, sparse=self.keep_sparse)
                        val_loss += loss.item() * len(x)
                        if self.n_ct is not None:
                            ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                            ct_logits.append(torch.softmax(ct_logit, dim=1))

                if self.n_ct is not None:
                    ct_truth = torch.cat(ct_truth)
                    ct_logits = torch.cat(ct_logits)
                    ct_pred = ct_logits.argmax(dim=1)
                    ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                    curr_metric = - val_loss / len(self.val_set)
                    self.logger("Time: {:.1f}s | Avg. Validation Loss: {:.2f} | CT Acc: {:.2f}".format(time.time() - start_time, val_loss / len(self.val_set), ct_acc))
                else:
                    curr_metric = - val_loss / len(self.val_set)
                    self.logger("Time: {:.1f}s | Avg. Validation Loss: {:.2f}".format(time.time() - start_time, val_loss / len(self.val_set)))
                
                if best_metric is None or curr_metric > best_metric:
                    torch.save(self.model.state_dict(), \
                        os.path.join(self.checkpoint_dir, "pretrain.pt"))
                    best_metric = curr_metric
                    self.logger("Model Saved!")
                
                self.model.train()
            
            self.logger("")
                
    def my_collate(self, batch):
        x = [item[0] for item in batch]
        y = torch.tensor([item[1] for item in batch])
        if len(batch[0]) == 3:
            ct = [item[2] for item in batch]
            return x, y, ct
        return x, y

class Config_eval:

    def __init__(self, data, data_loc, checkpoint_dir, checkpoint_name="best_model.pt", model="ProtoCell", \
         split_ratio = [0.5, 0.25, 0.25], batch_size = 4, h_dim = 128, z_dim = 32, n_layers = 2, \
         n_proto=8, device="cpu", seed=0, d_min=1, task=None, subsample=False, load_ct=True):


        assert len(split_ratio) == 3 and sum(split_ratio) == 1.0

        self.seed = seed
        self.data_loc = data_loc

        # load the target data
        if data.lower() == "lupus":
            assert task is not None
            dataset = load_lupus(task=task, load_ct=True)
            self.collate_fn = self.my_collate
        elif data.lower() == "cardio":
            dataset = load_cardio(load_ct=True)
        elif data.lower() == "covid":
            dataset = load_covid(load_ct=load_ct)
        elif data.lower() == "gausses":
            dataset = load_gausses(load_ct=load_ct, keep_sparse=self.keep_sparse)
        elif data.lower() == "cytof":
            dataset = load_cytof(data_dir=self.data_loc, load_ct=load_ct, keep_sparse=False, data_location = self.data_loc)
            # dataset = load_cytof(load_ct=load_ct, keep_sparse=self.keep_sparse)      
        else:
            raise ValueError("Data [:s] not supported".format(data))

        if load_ct:
            self.n_ct = len(dataset.ct_id)
        else:
            self.n_ct = None

        self.collate_fn = self.my_collate

        # split the dataset according to train_val_test_split
        self.train_set, self.test_set = train_test_split(dataset, test_size=split_ratio[2], random_state=self.seed)
        self.train_set, self.val_set = train_test_split(self.train_set, test_size=split_ratio[1] / (1-split_ratio[2]), random_state=self.seed)

        self.device = device
        self.batch_size = batch_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_classes = max(dataset.y) + 1
        self.n_proto = n_proto
        self.d_min = d_min
        self.ct_id = dataset.ct_id
        self.class_id = dataset.class_id

        self.model_type = model
        self.lambdas = {
            "lambda_1": 0,
            "lambda_2": 0,
            "lambda_3": 0,
            "lambda_4": 0,
            "lambda_5": 0,
            "lambda_6": 0
        }
                
        if self.model_type == "ProtoCell":
            self.model = ProtoCell(dataset.X[0].shape[1], self.h_dim, self.z_dim, self.n_layers, \
                self.n_proto, self.n_classes, self.lambdas, self.n_ct, self.device, d_min=self.d_min) 
        else:
            raise ValueError("Model [:s] not supported".format(self.model_type))

        self.model.to(self.device)

        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        if subsample:
            c_counts = []
            for ins in self.train_set:
                c_counts.append(ins[0].shape[0])
            W = sum(c_counts) // (5 * len(c_counts))

            train_set = []
            for ins in self.train_set:
                n = ins[0].shape[0]
                if n < W * 2:
                    train_set.append(ins)
                    continue
                elif n < W * 4:
                    D = round(2 * n / W) # number of times for subsampling
                else:
                    D = 8
                idx = np.arange(n)
                for _ in range(D):
                    np.random.shuffle(idx)
                    if self.n_ct is None:
                        train_set.append([ins[0][idx[:W]], ins[1]])
                    else:
                        train_set.append([ins[0][idx[:W]], ins[1], [ins[2][i] for i in idx[:W]]])
            print("Training Data Augmented: {:d} --> {:d}".format(len(self.train_set), len(train_set)))
            self.train_set = train_set
            self.train_batch_size = 8 * self.batch_size # 64 for the 625 setting
        else:
            self.train_batch_size = self.batch_size
        
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.eval()
    
    def reason(self, dataset="test"):
        if dataset == "train":
            loader = DataLoader(self.train_set, batch_size = self.batch_size, shuffle = False, collate_fn=self.collate_fn)
        elif dataset == "val":
            loader = DataLoader(self.val_set, batch_size = self.batch_size, shuffle = False, collate_fn=self.collate_fn)
        elif dataset == "test":
            #print("config_mk line 624 myles changed some here")
            #with open('all_data_hf_il1b_week0.pkl', 'rb') as file:   #Myles added
            #    all_data = pickle.load(file)                                 #Myles added
            #loader = DataLoader(all_data, batch_size = self.batch_size, shuffle = False, collate_fn=self.collate_fn) #Myles added
            loader = DataLoader(self.test_set, batch_size = self.batch_size, shuffle = False, collate_fn=self.collate_fn)

        self.Z = []
        self.CLOG = []
        self.LOG = []
        self.C2P = []
        self.Y = []
        self.IMP = []
        self.CT = []
        self.SPLIT = []
        with torch.no_grad():
            for x, y, ct in tqdm.tqdm(loader):
                # TMP.append(config.model(x,y,ct)[1])
                split_idx = [0]
                for i in range(len(x)):
                    split_idx.append(split_idx[-1]+x[i].shape[0])
                if len(self.SPLIT) == 0:
                    self.SPLIT += split_idx
                else:
                    self.SPLIT += [i+self.SPLIT[-1] for i in split_idx[1:]]
                
                # x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
                x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device) #myles
                z = self.model.encode(x)
                self.Z.append(z.detach())
                c2p_dists = torch.pow(z[:, None] - self.model.prototypes[None, :], 2).sum(-1)                
                import_scores = self.model.compute_importance(x)
                print("config_mk line 655 myles added")  #Myles added
                with open('import_scores_all_data.pkl', 'wb') as file:
                    pickle.dump(import_scores, file)
                c_logits = (1 / (c2p_dists+0.5))[:,None,:].matmul(import_scores).squeeze(1) # (n_cell, n_classes)
                with open('c_logits_all_data.pkl', 'wb') as file:
                    pickle.dump(c_logits, file)
                logits = torch.stack([c_logits[split_idx[i]:split_idx[i+1]].mean(dim=0) for i in range(len(split_idx)-1)])
                with open('logits_all_data.pkl', 'wb') as file:
                    pickle.dump(logits, file)
                self.CLOG.append(c_logits.detach())
                self.LOG.append(logits.detach())
                self.C2P.append(c2p_dists.detach())
                self.Y.append(y.to(self.device))
                self.IMP.append(import_scores.detach())
                self.CT.append([j for i in ct for j in i])

        self.Z = torch.cat(self.Z, dim=0)
        self.C2P = torch.cat(self.C2P, dim=0)
        self.Y = torch.cat(self.Y, dim=0)
        self.IMP = torch.cat(self.IMP, dim=0)
        self.CT = torch.tensor([j for i in self.CT for j in i]).to(self.device)
        self.CLOG = torch.cat(self.CLOG, dim=0)
        self.LOG = torch.cat(self.LOG, dim=0)
        self.prototype = self.model.prototypes.detach()

        print("Y accuracy:", ((self.LOG.argmax(1) == self.Y).sum() / len(self.Y)).cpu().item())

    def my_collate(self, batch):
        x = [item[0] for item in batch]
        y = torch.tensor([item[1] for item in batch])
        if len(batch[0]) == 3:
            ct = [item[2] for item in batch]
            return x, y, ct
        return x, y



def patid_identify(csv_file_name, train_data, val_data , test_data ):
    cell_count = pd.read_csv(csv_file_name)
    cell_count = cell_count.iloc[:,1:5]
    patient_list = list()
    print("\n training data set cell counts:")
    for ii in range(len(train_data)):
        c_count = train_data[ii][0].shape[0]
        pat_id = ((cell_count['patient_id'])[cell_count['count']==c_count]).iloc[0]
        pvo2 = (cell_count['disease'][cell_count['count']==c_count]).iloc[0]
        print("cell count: "+str(c_count)+", pat_id:"+str(pat_id)+"  "+pvo2)
        patient_list.append(pat_id)
    print("\n validation data set cell counts:")    
    for ii in range(len(val_data)):
        c_count = val_data[ii][0].shape[0]
        pat_id = ((cell_count['patient_id'])[cell_count['count']==c_count]).iloc[0]
        pvo2 = (cell_count['disease'][cell_count['count']==c_count]).iloc[0]
        print("cell count: "+str(c_count)+", pat_id:"+str(pat_id)+"  "+pvo2)
        patient_list.append(pat_id)
    print("\n test data set cell counts:")  
    for ii in range(len(test_data)):
        c_count = test_data[ii][0].shape[0]
        pat_id = ((cell_count['patient_id'])[cell_count['count']==c_count]).iloc[0]
        pvo2 = (cell_count['disease'][cell_count['count']==c_count]).iloc[0]
        print("cell count: "+str(c_count)+", pat_id:"+str(pat_id)+"  "+pvo2)
        patient_list.append(pat_id)
    return patient_list

def find_patid(csv_file_name, cell_count):
    cell_count_patid_data = pd.read_csv(csv_file_name)
    return ((cell_count_patid_data['patient_id'])[cell_count_patid_data['count']==cell_count]).iloc[0]