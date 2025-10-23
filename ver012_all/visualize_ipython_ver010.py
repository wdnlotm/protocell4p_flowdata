import torch
from utils_ver010 import *
from config_cytof_ver010 import *
import argparse
import os

def visualize_latent_sample(args, k=20, n_sample=None): # k: k cells that make most contribution for each sample's classification

    print("Checkpoint path:", os.path.join(args.checkpoint_dir,args.checkpoint_name))

    config = Config_eval(
        data = args.data,
        data_loc = args.data_loc,
        task = args.task,
        model = args.model,
        split_ratio = args.split_ratio,
        batch_size = args.batch_size,
        n_proto = args.n_proto,
        h_dim = args.h_dim,
        z_dim = args.z_dim,
        n_layers = args.n_layers,
        d_min = args.d_min,
        device = args.device,
        seed = args.seed,
        subsample = args.subsample,
        load_ct = args.load_ct,
        checkpoint_dir = args.checkpoint_dir,
        checkpoint_name = args.checkpoint_name
    )
    
    config.reason(dataset="test")

    if n_sample is not None:
        idx = torch.randperm(len(config.Y))[:n_sample]
    else:
        idx = torch.arange(len(config.Y))

    n_class = len(config.class_id)
    state_cont = [(n_class * config.CLOG[:,i] - config.CLOG.sum(1)) / (n_class - 1) for i in range(len(config.class_id))]
    # state_cont = [2 * config.CLOG[:,i] - config.CLOG.sum(1) for i in range(len(config.class_id))]
    for i in range(len(config.class_id)):
        state_cont[i] = torch.stack([config.SPLIT[j] + state_cont[i][config.SPLIT[j]:config.SPLIT[j+1]].topk(k)[1] for j in range(len(config.Y))])
        
    state_cont = torch.stack(state_cont) # n_class * n_sample * k
    
    if "checkpoint" in args.checkpoint_dir:
        save_dir = args.checkpoint_dir.replace("checkpoint", "case_study")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = args.checkpoint_dir
    
    s = 20

    for i in idx:
        X = config.Z[config.SPLIT[i]:config.SPLIT[i+1]].cpu().numpy()
        print(X.shape)
        label = config.CT[config.SPLIT[i]:config.SPLIT[i+1]].cpu().numpy().tolist()
        # label_text = config.ct_id
        label_text = None
        save_path = os.path.join(save_dir, str(i.item())+"_"+config.class_id[config.Y[i]]+".png")
        prototype = config.prototype.cpu().numpy()
        contribution = torch.stack([config.Z[state_cont[j,i,:]] for j in range(len(config.class_id))]).cpu().numpy()
        class_text = ["\n".join(j.split(" ")) for j in config.class_id]
        # TSNE plot for data in the latent space according to CT
        if args.data == "cardio" or args.data == "covid":
            plot_tsne_w_cont(X=X, label=label, prototype=prototype, contribution=contribution, class_text=class_text, \
                label_text=label_text, save_path=save_path, s=s, pattern="Paired")
        else:
            plot_tsne_w_cont(X=X, label=label, prototype=prototype, contribution=contribution, class_text=class_text, \
                label_text=label_text, save_path=save_path, s=s, pattern="Paired")
            # plot_tsne_w_cont(X=X, label=label, prototype=prototype, contribution=contribution, class_text=class_text, \
            #     label_text=label_text, save_path=save_path, s=s)   #Original. myles changed
            
        print("Complete: TSNE plot for sample {:d} in the latent space".format(i.item()))

def visualize_latent(args, k=10): # k: k cells that make most contribution for each sample's classification

    print("Checkpoint directory:", args.checkpoint_dir)

    config = Config_eval(
        data = args.data,
        task = args.task,
        model = args.model,
        split_ratio = args.split_ratio,
        batch_size = args.batch_size,
        n_proto = args.n_proto,
        h_dim = args.h_dim,
        z_dim = args.z_dim,
        n_layers = args.n_layers,
        d_min = args.d_min,
        device = args.device,
        seed = args.seed,
        subsample = args.subsample,
        load_ct = args.load_ct,
        checkpoint_dir = args.checkpoint_dir
    )
    
    config.reason(dataset="test")

    state_cont = [2 * config.CLOG[:,i] - config.CLOG.sum(1) for i in range(len(config.class_id))]
    for i in range(len(config.class_id)):
        state_cont[i] = torch.stack([config.SPLIT[j] + state_cont[i][config.SPLIT[j]:config.SPLIT[j+1]].topk(k)[1] for j in range(len(config.Y))])

    state_cont = torch.stack(state_cont) # n_class * n_sample * k
    state_cont = state_cont.reshape(len(config.class_id), -1)

    save_dir = args.checkpoint_dir

    X = config.Z.cpu().numpy()
    label = config.CT.cpu().numpy()
    # label_text = config.ct_id
    label_text = None
    save_path = os.path.join(save_dir, "latent_tsne_test.png")
    prototype = config.prototype.cpu().numpy()
    contribution = torch.stack([config.Z[state_cont[i]] for i in range(len(config.class_id))]).cpu().numpy()
    class_text = config.class_id

    s = 3

    # TSNE plot for data in the latent space according to CT
    plot_tsne_w_cont(X=X, label=label, prototype=prototype, contribution=contribution, class_text=class_text, \
        label_text=label_text, save_path=save_path, s=s)
    print("Complete: TSNE plot for test data in the latent space")