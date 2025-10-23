import torch
import torch.nn as nn
import os

class Logger4model:
  
    def __init__(self, path, log=True):
        self.path = path
        self.log = log

    def __call__(self, content, **kwargs):
        # print(content, **kwargs)
        if self.log:
            with open(self.path, 'a') as f:
                print(content, file=f, **kwargs)


class ProtoCell(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, n_layers, n_proto, n_classes, lambdas,\
                 n_ct=None, device="cpu", d_min=1, log_dir = "../log", log_file = "log2_testing.txt"):
        super(ProtoCell, self).__init__()
        # print('hi from model top') 

        self.log_file = log_file
        self.log_file = self.log_file.replace('.txt', '_model.txt')
        self.log_dir = log_dir

        print("\n The log file name is " + self.log_file)
        print("\n The log file is saved in "+self.log_dir)
        
        self.logger = Logger4model(os.path.join(self.log_dir, self.log_file))
        # self.logger = Logger4model(os.path.join("../log", "log2_testing.txt"))
        
        self.device = device

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_proto = n_proto
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_ct = n_ct
        self.d_min = d_min
        
        assert self.n_layers > 0

        self.lambda_1 = lambdas["lambda_1"]
        self.lambda_2 = lambdas["lambda_2"]
        self.lambda_3 = lambdas["lambda_3"]
        self.lambda_4 = lambdas["lambda_4"]
        self.lambda_5 = lambdas["lambda_5"]
        self.lambda_6 = lambdas["lambda_6"]

        self.enc_i = nn.Linear(self.input_dim, self.h_dim)
        self.enc_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.enc_z = nn.Linear(self.h_dim, self.z_dim)

        self.dec_z = nn.Linear(self.z_dim, self.h_dim)
        self.dec_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.dec_i = nn.Linear(self.h_dim, self.input_dim)

        self.imp_i = nn.Linear(self.input_dim, self.h_dim)
        self.imp_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.imp_p = nn.Linear(self.h_dim, self.n_proto * self.n_classes)

        self.activate = nn.LeakyReLU()

        self.prototypes = nn.parameter.Parameter(torch.empty(self.n_proto, self.z_dim), requires_grad = True)

        self.ce_ = nn.CrossEntropyLoss(reduction="mean")

        nn.init.xavier_normal_(self.enc_i.weight)
        nn.init.xavier_normal_(self.enc_z.weight)
        nn.init.xavier_normal_(self.dec_z.weight)
        nn.init.xavier_normal_(self.dec_i.weight)
        nn.init.xavier_normal_(self.imp_i.weight)
        nn.init.xavier_normal_(self.imp_p.weight)
        nn.init.xavier_normal_(self.prototypes)
        for i in range(self.n_layers - 1):
            nn.init.xavier_normal_(self.enc_h[i].weight)
            nn.init.xavier_normal_(self.dec_h[i].weight)
            nn.init.xavier_normal_(self.imp_h[i].weight)

        if self.n_ct is not None:
            self.ct_clf1 = nn.Linear(self.n_proto, self.n_ct)
            self.ct_clf2 = nn.Linear(self.n_proto * self.n_classes, self.n_ct)
            nn.init.xavier_normal_(self.ct_clf1.weight)
            nn.init.xavier_normal_(self.ct_clf2.weight)

    def forward(self, x, y, ct=None, sparse=True, epoch_num = 77):   #myles added epoch_num
        #print('Executing forward')  
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)

        z = self.encode(x)
        
        import_scores = self.compute_importance(x) # (n_cell, n_proto, n_class)
        
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        c_logits = (1 / (c2p_dists+0.5))[:,None,:].matmul(import_scores).squeeze(1) # (n_cell, n_classes)
        logits = torch.stack([c_logits[split_idx[i]:split_idx[i+1]].mean(dim=0) for i in range(len(split_idx)-1)])

        clf_loss = self.ce_(logits, y)

        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf2(import_scores.reshape(-1, self.n_proto * self.n_classes))
            ct_loss = self.ce_(ct_logits, torch.tensor([j for i in ct for j in i]).to(self.device))            
        else:
            ct_loss = 0

        total_loss = clf_loss + self.lambda_6 * ct_loss
        self.logger("[Epoch {:d}] from forward z clf_loss z {:.3f}".format(epoch_num, clf_loss))
        self.logger("[Epoch {:d}] from forward z ct_loss z {:.3f}".format(epoch_num, ct_loss))
        # print(self.log_file)
        # print("line 110 myles prints clf_loss z {:.2f}".format(clf_loss))
        # self.logger("line 110 myles prints ct_loss z {:.2f}".format(clf_loss))
        # print(epoch_num)  

        if ct is not None:
            return total_loss, logits, ct_logits, clf_loss, ct_loss
        return total_loss, logits
    
    def encode(self, x):
        #print('Executing encode')  
        h_e = self.activate(self.enc_i(x))
        for i in range(self.n_layers - 1):
            h_e = self.activate(self.enc_h[i](h_e))
        z = self.activate(self.enc_z(h_e))
        return z

    def decode(self, z):
        #print('Executing decode')  
        h_d = self.activate(self.dec_z(z))
        for i in range(self.n_layers - 1):
            h_d = self.activate(self.dec_h[i](h_d))
        x_hat = torch.relu(self.dec_i(h_d))
        return x_hat

    def compute_importance(self, x):
        #print('Executing compute_importance')  
        h_i = self.activate(self.imp_i(x))
        for i in range(self.n_layers - 1):
            h_i = self.activate(self.imp_h[i](h_i))
        import_scores = torch.sigmoid(self.imp_p(h_i)).reshape(-1, self.n_proto, self.n_classes)
        return import_scores
    
    def pretrain(self, x, y, ct=None, sparse=True, epoch_num = 77):
        #print('Executing pretrain')  
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)
        if ct is not None:
            ct = torch.tensor([j for i in ct for j in i]).to(self.device)
        z = self.encode(x)
        x_hat = self.decode(z)
        
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        if ct is None:
            p2c_dists = torch.pow(self.prototypes[:, None] - z[None, :], 2).sum(-1)
        else:
            p2c_dists = torch.stack([torch.pow(self.prototypes[:, None] - z[ct == t][None, :], 2).sum(-1).mean(-1) for t in ct.unique().tolist()]).T # n_proto * n_ct
        p2p_dists = (torch.pow(self.prototypes[:, None] - self.prototypes[None, :], 2).sum(-1)+1e-16).sqrt()

        recon_loss = (x - x_hat).pow(2).mean()
        c2p_loss = (c2p_dists).min(dim=1)[0].mean()
        p2c_loss = (p2c_dists + (torch.ones_like(p2c_dists).uniform_() < 0.3) * 1e9).min(dim=1)[0].mean()
        p2p_loss = ((self.d_min - p2p_dists > 0) * (self.d_min - p2p_dists)).pow(2).sum() / (self.n_proto * self.n_proto - self.n_proto)

        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf1(1 / (c2p_dists+0.5))
            ct_loss = self.ce_(ct_logits, ct)
        else:
            ct_loss = 0

        total_loss = self.lambda_1 * recon_loss +\
                     self.lambda_2 * c2p_loss +\
                     self.lambda_3 * p2c_loss +\
                     self.lambda_4 * p2p_loss +\
                     self.lambda_5 * ct_loss
        self.logger("[Epoch {:d}] from pretrain z recon_loss z {:.3f} ".format(epoch_num, recon_loss))
        #print("myles prints recon_loss z " + str(recon_loss.detach().cpu().numpy()))
        self.logger("[Epoch {:d}] from pretrain z c2p_loss z {:.3f} ".format(epoch_num, c2p_loss))
        #print("myles prints c2p_loss z " + str(c2p_loss.detach().cpu().numpy()))
        self.logger("[Epoch {:d}] from pretrain z p2c_loss z {:.3f} ".format(epoch_num, p2c_loss))
        #print("myles prints p2c_loss z " + str(p2c_loss.detach().cpu().numpy()))
        self.logger("[Epoch {:d}] from pretrain z p2p_loss z {:.3f} ".format(epoch_num, p2p_loss))
        #print("myles prints p2p_loss z " + str(p2p_loss.detach().cpu().numpy()))
        self.logger("[Epoch {:d}] from pretrain z ct_loss z {:.3f} ".format(epoch_num, ct_loss))
        #print("myles prints ct_loss z " + str(ct_loss.detach().cpu().numpy()))
        
        #print(ct)  #Myles added. Okay. ct is not none. It comes from load_data
        if ct is not None:
            #print('hi in pretrain')                          #Okay, it's printed
            #print(ct_logits) # Myles added.      #Okay, it's printed
            return total_loss, ct_logits   #myles add c2p_loss
        return total_loss

class BaseModel(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, n_layers, n_proto, n_classes, lambdas, n_ct=None, device="cpu", d_min=1):
        super(BaseModel, self).__init__()

        self.device = device

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_proto = n_proto
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_ct = n_ct
        self.d_min = d_min
        
        assert self.n_layers > 0

        self.lambda_1 = lambdas["lambda_1"]
        self.lambda_2 = lambdas["lambda_2"]
        self.lambda_3 = lambdas["lambda_3"]
        self.lambda_4 = lambdas["lambda_4"]
        self.lambda_5 = lambdas["lambda_5"]
        self.lambda_6 = lambdas["lambda_6"]

        self.enc_i = nn.Linear(self.input_dim, self.h_dim)
        self.enc_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.enc_z = nn.Linear(self.h_dim, self.z_dim)

        self.dec_z = nn.Linear(self.z_dim, self.h_dim)
        self.dec_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.dec_i = nn.Linear(self.h_dim, self.input_dim)

        self.clf = nn.Linear(self.n_proto, self.n_classes, bias=False)

        self.activate = nn.LeakyReLU()

        self.prototypes = nn.parameter.Parameter(torch.empty(self.n_proto, self.z_dim), requires_grad = True)

        self.ce_ = nn.CrossEntropyLoss(reduction="mean")

        nn.init.xavier_normal_(self.enc_i.weight)
        nn.init.xavier_normal_(self.enc_z.weight)
        nn.init.xavier_normal_(self.dec_z.weight)
        nn.init.xavier_normal_(self.dec_i.weight)
        nn.init.xavier_normal_(self.prototypes)
        nn.init.xavier_normal_(self.clf.weight)
        for i in range(self.n_layers - 1):
            nn.init.xavier_normal_(self.enc_h[i].weight)
            nn.init.xavier_normal_(self.dec_h[i].weight)

        if self.n_ct is not None:
            self.ct_clf1 = nn.Linear(self.n_proto, self.n_ct)
            nn.init.xavier_normal_(self.ct_clf1.weight)

    def forward(self, x, y, ct=None, sparse=True):
        
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)

        z = self.encode(x)
        
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        c_logits = self.clf(1 / (c2p_dists+0.5))
        logits = torch.stack([c_logits[split_idx[i]:split_idx[i+1]].mean(dim=0) for i in range(len(split_idx)-1)])

        clf_loss = self.ce_(logits, y)

        ct_loss = 0

        total_loss = clf_loss + self.lambda_6 * ct_loss

        if ct is not None:
            ct_logits = torch.ones(len(x), self.n_ct).to(self.device)
            return total_loss, logits, ct_logits    
        return total_loss, logits
    
    def encode(self, x):
        h_e = self.activate(self.enc_i(x))
        for i in range(self.n_layers - 1):
            h_e = self.activate(self.enc_h[i](h_e))
        z = self.activate(self.enc_z(h_e))
        return z

    def decode(self, z):
        h_d = self.activate(self.dec_z(z))
        for i in range(self.n_layers - 1):
            h_d = self.activate(self.dec_h[i](h_d))
        x_hat = torch.relu(self.dec_i(h_d))
        return x_hat
    
    def pretrain(self, x, y, ct=None, sparse=True):
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)
        if ct is not None:
            ct = torch.tensor([j for i in ct for j in i]).to(self.device)        
        z = self.encode(x)
        x_hat = self.decode(z)
        
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        if ct is None:
            p2c_dists = torch.pow(self.prototypes[:, None] - z[None, :], 2).sum(-1)
        else:
            p2c_dists = torch.stack([torch.pow(self.prototypes[:, None] - z[ct == t][None, :], 2).sum(-1).mean(-1) for t in ct.unique().tolist()]).T # n_proto * n_ct
        p2p_dists = (torch.pow(self.prototypes[:, None] - self.prototypes[None, :], 2).sum(-1)+1e-16).sqrt()

        recon_loss = (x - x_hat).pow(2).mean()
        c2p_loss = (c2p_dists).min(dim=1)[0].mean()
        p2c_loss = (p2c_dists + (torch.ones_like(p2c_dists).uniform_() < 0.3) * 1e9).min(dim=1)[0].mean()
        p2p_loss = ((self.d_min - p2p_dists > 0) * (self.d_min - p2p_dists)).pow(2).sum() / (self.n_proto * self.n_proto - self.n_proto)
        
        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf1(1 / (c2p_dists+0.5))
            ct_loss = self.ce_(ct_logits, ct)
        else:
            ct_loss = 0

        total_loss = self.lambda_1 * recon_loss +\
                     self.lambda_2 * c2p_loss +\
                     self.lambda_3 * p2c_loss +\
                     self.lambda_4 * p2p_loss +\
                     self.lambda_5 * ct_loss
        
        if self.n_ct is not None:
            return total_loss, ct_logits
        return total_loss