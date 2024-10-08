import click
import torch
import logging
import random
import numpy as np
import pandas as pd
import csv
import sys
import os
from sklearn import preprocessing
import math
import torch.nn.functional as F
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD_Dkmeans import DeepSVDD
from deepSVDD_Dkmeans import AE
from datasets.main import load_dataset
from networks.mine_soft_assign import soft_assign
from torch.autograd import Variable
from sklearn.cluster import KMeans
import os, shutil
import ipdb

################################################################################
# Settings
################################################################################
data_name = "toy"
dim_covs = 5

deg = 8
num_train_samples = 200


n_cluster = 10
net_name = "DCC"
load_model = None 
deep_main = True # use deep kmeans for xi or not. If False, f_W(xi) = xi

@click.command()
@click.option('--dataset_name', type=click.Choice(['knapsack','shortest_path','toy']),default=data_name)
@click.option('--dim_covs', type=int, default=dim_covs)
@click.option('--deg', type=int, default=deg)
@click.option('--num_train_samples', type=int, default=num_train_samples)
@click.option('--net_name', type=click.Choice(['DDDRO','DCC','IDCC']),default=net_name)
@click.option('--deep_main', type=bool, default=deep_main)

# @click.argument('side_info_path', type=click.Path(exists=True),default=None)
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
              
#@click.option('--test_path', type=click.Path(exists=True), default=None,
#              help='Test file path.')              
#@click.option('--train_path', type=click.Path(exists=True), default=None,
#              help='Test file path.')              
              
@click.option('--load_model', type=click.Path(exists=False), default=load_model,
              help='Model file path (default: None).')
@click.option('--n_cluster', type=int, default=n_cluster,
                help='Select the number of clusters')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')

@click.option('--beta', type=float, default=0.1, help='conditional network assignment control variable.')
@click.option('--lmbda', type=float, default=1, help='distance scaling factor for soft kmeans loss')
@click.option('--alpha', type=float, default=0.1, help='weight for the conditional loss')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=(200,), multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=(200,), multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')

def main(dataset_name, dim_covs, deg, num_train_samples,net_name,deep_main, load_config, load_model,n_cluster, objective, beta,lmbda, alpha, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    data_path = "../../../data/"+dataset_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
    test_path = "../../../data/"+dataset_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

    #plot_cov_dim = 1
    #plot_path = "../../../data/"+data_name+"/"+str(dim_covs)+"/"+str(deg)+"/plot/"+str(plot_cov_dim)+"/"
    plot_path = None

    pretrained_model = data_path+net_name+"/"+str(n_cluster)+"/AE.pt"

    


    # Get configuration
    cfg = Config(locals().copy())

    deep_main = cfg.settings['deep_main']

    result = cfg.settings['data_path']+net_name+"/"+str(n_cluster)+"/"
    
    os.makedirs(result, exist_ok=True)

    # if the result already exists, then end
    if os.path.exists(result+'train_assignments.npy') and os.path.exists(result+'test_assignments.npy'):
        return
        

    lr=cfg.settings['lr']
    n_classes=cfg.settings['n_cluster']
    alpha=cfg.settings['alpha']
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Print arguments

    logger.info('Data path is %s.' % data_path)


    logger.info('Dataset: %s' % dataset_name)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    
    
    
    # Load data
    #separate side info from the data
    # data=pd.read_csv(data_path, sep=",", header=None)
    # side_info_train = data.iloc[:, 0:2]
    # data.iloc[:, 2:].to_csv(data_path,index=False,header=False)
    

    dataset = load_dataset(data_path, test_path,plot_path)
    cfg.settings['batch_size'] = dataset.train_set.side_train_set.size(0)
    batch_size = dataset.train_set.side_train_set.size(0)
    side_dim=dataset.train_set.side_train_set.size(1)
    main_dim=dataset.train_set.data_train_set.size(1)
    logger.info('input dim of network is: %s'%dataset.train_set.data_train_set.size(0))
    out_dim=3 #3 for simulated
    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'],cfg.settings['n_cluster'], cfg.settings['beta'], cfg.settings['alpha'],cfg.settings['nu'])
    k_means_AE = AE(net_name,main_dim,side_dim,out_dim,cfg.settings['objective'],cfg.settings['n_cluster'], cfg.settings['beta'], cfg.settings['alpha'],cfg.settings['nu'])
    deep_SVDD.set_network(net_name,main_dim,side_dim,out_dim,cfg.settings['beta'],cfg.settings['lmbda'],cfg.settings['n_cluster'])


    logger.info('cond_network: %s' % deep_SVDD.encoder)
    logger.info('cond_network: %s' % deep_SVDD.decoder)
    logger.info('cond_network: %s' % deep_SVDD.soft_KMeans)
    logger.info('main_network: %s' % deep_SVDD.net_main)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # check if there is a pretrained model file
        if os.path.exists(cfg.settings['pretrained_model']):
            logger.info('Pretrained model exists. Loading...')
            deep_SVDD.load_model(model_path=cfg.settings['pretrained_model'], load_ae=True)
        else:
            # Log pretraining details
            logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
            logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
            logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
            logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
            logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
            logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

            # Pretrain model on dataset (via autoencoder)
            deep_SVDD.pretrain(dataset,
                            optimizer_name=cfg.settings['ae_optimizer_name'],
                            lr=cfg.settings['ae_lr'],
                            n_epochs=cfg.settings['ae_n_epochs'],
                            lr_milestones=cfg.settings['ae_lr_milestone'],
                            batch_size=cfg.settings['ae_batch_size'],
                            weight_decay=cfg.settings['ae_weight_decay'],
                            device=device,
                            n_jobs_dataloader=n_jobs_dataloader)
        
            # save the pretrained model
            deep_SVDD.save_model(export_model=cfg.settings['pretrained_model'], save_ae=True)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    

    
    # Train model on dataset
    # logger.info(assignments)
    k_means_AE.encoder = deep_SVDD.encoder
    k_means_AE.decoder = deep_SVDD.decoder
    centroids = Variable(torch.tensor(k_means_AE.centroid_init(dataset,n_classes, out_dim,batch_size)))
    # logger.info(centroids)
    for _ in range(100):
        centroids, assignments = k_means_AE.train(dataset,centroids,batch_size)
        # logger.info(centroids)
        # logger.info(torch.sum(assignments,axis=0))

    list_1=assignments
    
    test_batch_size = dataset.test_set.side_test_set.size(0)

    with torch.no_grad():
        _, test_loader = dataset.loaders(batch_size=test_batch_size, num_workers=n_jobs_dataloader)
        for data in test_loader:
            side,_,_ = data
            # logger.info('embeddings: %s'%embeddings)
            embeddings = k_means_AE.encoder(side.to(device))
            _, cluster_assignments = k_means_AE.KMeans(embeddings, centroids)

        # logger.info(test_label)
        test_assignments = np.zeros((len(test_loader.dataset), n_cluster))
        for i in range(test_assignments.shape[0]):
            test_assignments[i, cluster_assignments[i]] = 1

        if plot_path is not None:
            plot_assignments = test_assignments[-num_plot_samples:,:]
            test_assignments = test_assignments[:-num_plot_samples,:]

        np.save(result+'train_assignments', list_1.numpy())
        np.save(result+'test_assignments', test_assignments)
        if plot_path is not None:
            np.save(plot_path+net_name+"_"+str(num_train_samples)+"_"+str(n_cluster)+"_"+"plot_assignments.npy", plot_assignments.numpy())
    
    num_samples = []
    for k in range(n_cluster):
        num_samples.append(torch.sum(list_1[:,k]))

    if deep_main:
        for k in range(n_classes):
            logger.info('Cluster : %g' %k)
            deep_SVDD.set_network(net_name,main_dim,side_dim,out_dim,cfg.settings['beta'],cfg.settings['lmbda'],cfg.settings['n_cluster'])
            deep_SVDD.train(k,list_1,dataset,
                        optimizer_name=cfg.settings['optimizer_name'],
                        lr=cfg.settings['lr'],
                        n_epochs=cfg.settings['n_epochs'],
                        lr_milestones=cfg.settings['lr_milestone'],
                        batch_size=cfg.settings['batch_size'],
                        weight_decay=cfg.settings['weight_decay'],
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader)
            # logger.info(deep_SVDD.trainer.assignment.numpy())

            # logger.info(itertools.islice(deep_SVDD.net_main.state_dict(), 2))
            
            
            #normalize
            n_samples = 0
            c = torch.zeros(deep_SVDD.net_main.rep_dim,device=device)
            norm_data=pd.DataFrame()
            deep_SVDD.net_main.eval()
            n_samples=0
            cov=[0]
            with torch.no_grad():
                train_loader, _ = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
                for data in train_loader:
                    # get the inputs of the batch
                    side,inputs, _ = data
                    inputs_k=inputs[np.argmax(list_1,axis=1)==k,:].clone().detach()
                    inputs_k = inputs_k.to(device)

                    outputs = deep_SVDD.net_main(inputs_k)
                    n_samples = inputs_k.shape[0]
                    c = torch.sum(outputs, dim=0)
                    cov=torch.cov(outputs.T)
                    

            # logger.info(c)
            c /= n_samples
            c[(abs(c) < 0.01) & (c < 0)] = -0.01
            c[(abs(c) < 0.01) & (c >= 0)] = 0.01


            with open(result+'c_'+str(k)+'.txt', 'w') as f:
                f.write(','.join(str(i) for i in deep_SVDD.c.numpy()))
            
            j=0
            # logger.info(deep_SVDD.net.state_dict())
            for item in deep_SVDD.net.state_dict():
                with open(result+'W_'+str(int(k))+'_'+str(j)+'.txt', 'w') as f:
                    for l in deep_SVDD.net.state_dict()[item]:
                        f.write(','.join(str(i) for i in l.numpy()))
                        f.write('\n')
                    j+=1
                        
            #np.set_printoptions(threshold=sys.maxsize)
            # print(cov)
            # convert cov to a numpy array
            cov_np = cov.numpy()
            # if cov_np has a large condition number, then add a small number to the diagonal
            if np.linalg.cond(cov_np) > 1/sys.float_info.epsilon:
                cov_np += 0.001*np.eye(cov_np.shape[0])

            with open(result+'cov_'+str(k)+'.txt', 'w') as f:
                for item in cov:
                    f.write(','.join(str(i) for i in item.numpy()))
                    f.write('\n')
            # save model
            deep_SVDD.save_model(export_model=result+"model_"+str(k)+".pt", save_ae=True)

        # load covs.csv from plot_path
        if plot_path is not None:
            covs = pd.read_csv(plot_path+'covs.csv', header=None).to_numpy()
            num_plot_samples = covs.shape[0]
        else:
            num_plot_samples = 0        
        
    


if __name__ == '__main__':
    main()
