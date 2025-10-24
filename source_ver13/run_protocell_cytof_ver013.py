from config_cytof_ver013 import Config
import torch

#input_arg has arguments_for_protocell and arguments_for_visualize
from input_args import *

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H_%M_%S")
import datetime
today = datetime.date.today()

global seed_numb
seed_numb = 123

## Protype counts. usually = cluster count
pt_numb = 10 

## hidden layer dimension
h_dime = 16

## latent space dimension
z_dime = 8

#########  Data split, [split_1: train, split_2:val, split_3:test]
split_1 = round(0.62, 4)
split_2 = round( (1-split_1)/2, 5)
split_3 = round(1 - split_1 - split_2, 6)
split_ratio_input = [split_1, split_2, split_3]


log_file_name = 'log_' + str(today) + '_' + str(current_time) + "_" + str(seed_numb) + ".txt"
print(log_file_name)

####### Data folder
data_folder_name = 'bcell_50mil'
data_loc_str = '../data/'+ data_folder_name

exp_string1 = 'Prt_cytof_'+ data_folder_name 
exp_string2 = '_' + str(pt_numb) + 'pt_' + 'zd'+str(z_dime)+'_' + str(int(round(split_1*100,0)))  +'_spt1_sd' + str(seed_numb)
exp_string =  exp_string1 + exp_string2
print(exp_string)

### loss function weight. lambda5, 6 are related to ct_loss
lamb = 1.0

#lambda_5 and lambda_6 are related to load_ct
args = arguments_for_protocell(data = "cytof",
                               data_loc = data_loc_str,
                               model = "ProtoCell",
                               seed = seed_numb,
                               exp_str = exp_string,
                               log_file_name = log_file_name,
                               h_dim = h_dime,
                               z_dim = z_dime,
                               n_layers = 2, 
                               n_proto = pt_numb,
                               test_step = 21,
                               split_ratio = split_ratio_input, #[0.5, 0.25, 0.25] #0.70834 + 0.04166  +0.25
                               max_epoch = 101,
                               batch_size = 500,
                               subsample = False,
                               eval = None,
                               pretrained = False, #False is good
                               max_epoch_pretrain = 30,
                               keep_sparse = False,
                               lr = 0.0005,
                               device = "cuda:0", #"cpu", #device = "cuda:0",
                               load_ct = True,
                               lambda_1 = lamb, lambda_2 = lamb, lambda_3 = lamb, 
                               lambda_4 = lamb, lambda_5 = lamb, lambda_6 = lamb)

torch.manual_seed(seed_numb)
torch.cuda.manual_seed(seed_numb)

config = Config(
        data = args.data,
        data_loc = args.data_loc,
        log_file_name = args.log_file_name,
        model = args.model,
        split_ratio = args.split_ratio,
        lr = args.lr,
        max_epoch = args.max_epoch,
        batch_size = args.batch_size,
        test_step = args.test_step,
        h_dim = args.h_dim,
        z_dim = args.z_dim,
        n_layers = args.n_layers,
        n_proto = args.n_proto,
        device = args.device,
        seed = args.seed,
        exp_str = args.exp_str,
        task = args.task,
        subsample = False if args.subsample is None else args.subsample,
        eval = False if args.eval is None else args.eval,
        load_ct = False if args.load_ct is None else args.load_ct,
        d_min = args.d_min,
        lambda_1 = args.lambda_1,
        lambda_2 = args.lambda_2,
        lambda_3 = args.lambda_3,
        lambda_4 = args.lambda_4,
        lambda_5 = args.lambda_5,
        lambda_6 = args.lambda_6,
        keep_sparse = False if args.keep_sparse is None else args.keep_sparse,
        pretrained = False if args.pretrained is None else args.pretrained,
        lr_pretrain = args.lr_pretrain,
        max_epoch_pretrain = args.max_epoch_pretrain
)

config.train()
