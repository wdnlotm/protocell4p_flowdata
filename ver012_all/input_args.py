#Object to set the model parameters
class arguments_for_protocell:
    def __init__(self,
                 model,
                 exp_str,   #type=str, help="special string to identify an experiment"
                 data = "PBMC", #type = string
                 data_loc = "../data/",
                 log_file_name = "log.txt",
                 split_ratio = [0.7, 0.15, 0.15],
                 lr = 1e-4,
                 max_epoch = 50,
                 batch_size = 256,
                 test_step = 1, 
                 h_dim = 128,
                 z_dim = 32,
                 n_layers = 2, 
                 n_proto = 20, 
                 device = "cuda:0", 
                 seed = None, 
                 task = None,
                 subsample = False, 
                 eval = False, 
                 load_ct = False, 
                 d_min = 1.0, 
                 keep_sparse = True,
                 pretrained = False,
                 lr_pretrain = 1e-2,
                 max_epoch_pretrain = 20,
                 lambda_1 = 1.0, lambda_2 = 1.0, lambda_3 = 1.0, 
                 lambda_4 = 1.0, lambda_5 = 1.0, lambda_6 = 1.0):
        self.model = model
        self.exp_str = exp_str
        self.data = data
        self.data_loc = data_loc
        self.log_file_name = log_file_name
        self.split_ratio = split_ratio
        self.lr = lr 
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.test_step = test_step
        self.h_dim = h_dim
        self.z_dim = z_dim 
        self.n_layers = n_layers
        self.n_proto = n_proto
        self.device = device
        self.seed = seed
        self.task = task
        self.subsample = subsample
        self.eval = eval
        self.load_ct = load_ct
        self.d_min = d_min
        self.keep_sparse = keep_sparse
        self.pretrained = pretrained 
        self.lr_pretrain = lr_pretrain
        self.max_epoch_pretrain = max_epoch_pretrain
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.lambda_5 = lambda_5
        self.lambda_6 = lambda_6 



class arguments_for_visualize:
    def __init__(self,
                 data = "lupus", 
                 data_loc = "../data/",
                 task = None,
                 model = "ProtoCell",
                 split_ratio = [0.5, 0.25, 0.25],
                 batch_size = 500,
                 n_proto = 20, 
                 h_dim = 128,
                 z_dim = 32,
                 n_layers = 2,
                 d_min = 1.0,
                 gamma = 0.1,
                 device = "cuda:0", 
                 seed = 123,
                 subsample = False, 
                 load_ct = True,
                 keep_sparse = False,
                 checkpoint_dir = "location",
                 checkpoint_name = "best_model.pt",
                 type = "total",
                 n_sample = None,
                 k = 10):
        self.data = data
        self.data_loc = data_loc
        self.task = task
        self.model = model
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.n_proto = n_proto
        self.h_dim = h_dim
        self.z_dim = z_dim 
        self.n_layers = n_layers
        self.d_min = d_min
        self.gamma = gamma        
        self.device = device
        self.seed = seed
        self.subsample = subsample
        self.load_ct = load_ct
        self.keep_sparse = keep_sparse
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.type = type
        self.n_sample = n_sample
        self.k = k