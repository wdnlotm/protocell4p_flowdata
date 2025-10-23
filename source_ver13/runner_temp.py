seed_numb = 123
from config_cytof_ver010 import Config
import torch

#input_arg has arguments_for_protocell and arguments_for_visualize
from input_args import *   

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H_%M_%S")
import datetime
today = datetime.date.today()

log_file_name = 'log_' + str(today) + '_' + str(current_time) + "_" + str(seed_numb) + ".txt"
print(log_file_name)

data_folder_name = 'CTA_Tcell_CAC_cut_at_105'
data_loc_str = '../data/'+ data_folder_name

exp_string = 'protocell_cytof_'+ data_folder_name + '_55_' + str(seed_numb)

lamb = 1.0

#lambda_5 and lambda_6 are related to load_ct
args = arguments_for_protocell(data = "cytof",
                               data_loc = data_loc_str,
                               model = "ProtoCell",
                               seed = seed_numb,
                               exp_str = exp_string,
                               log_file_name = log_file_name,
                               h_dim = 16,
                               z_dim = 8,
                               n_layers = 2, 
                               n_proto = 12,
                               test_step = 10,
                               split_ratio = [0.55, 0.225, 0.225], #[0.5, 0.25, 0.25] #0.70834 + 0.04166  +0.25
                               max_epoch = 3000,
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

