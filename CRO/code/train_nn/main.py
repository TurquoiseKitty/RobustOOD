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

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD_new import DeepSVDD
from datasets.main import load_dataset
from networks.mine_soft_assign import soft_assign

    

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'mine','portfolio']),default='portfolio')
@click.argument('net_name', type=click.Choice(['port_soft_assign','soft_assign','mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'mine_net2', 'mine_net3', 'mine_net4', 'mine_net5', 'mine_sp','mine_gen','port_soft_assign_AE3']),default='port_soft_assign_AE3')
@click.argument('xp_path', type=click.Path(exists=True),default = r'..\..\path\xp')
@click.argument('data_path', type=click.Path(exists=True),default = r'..\..\path\data\finance\final_port.csv')
@click.argument('test_path', type=click.Path(exists=True),default = r'..\..\path\data\finance\final_port.csv')
# @click.argument('side_info_path', type=click.Path(exists=True),default=None)
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
              
#@click.option('--test_path', type=click.Path(exists=True), default=None,
#              help='Test file path.')              
#@click.option('--train_path', type=click.Path(exists=True), default=None,
#              help='Test file path.')              
              
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--n_clusters', type=int, default=1,
                help='Select the number of clusters')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--beta', type=float, default=0.1, help='conditional network assignment control variable.')
@click.option('--alpha', type=float, default=0.1, help='weight for the conditional loss')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=(9900,), multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=False,
              help='Pretrain neural network parameters via autoencoder.') 
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=False,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, test_path,load_config, load_model,n_clusters, objective, beta, alpha, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
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
    dataset = load_dataset(dataset_name, data_path, test_path, xp_path)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'],cfg.settings['n_clusters'], cfg.settings['beta'], cfg.settings['alpha'],cfg.settings['nu'])
    deep_SVDD.set_network(net_name,cfg.settings['n_clusters'])
    logger.info('cond_network: %s' % deep_SVDD.net_cond)
    logger.info('main_network: %s' % deep_SVDD.net_main)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
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

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    

    
    # Train model on dataset
    # logger.info(assignments)
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

   
    
    result= r'path/simulated_2clus'
    os.makedirs(result, exist_ok=True)
    
    #normalize
    n_samples = 0
    c = torch.zeros(deep_SVDD.net_main.rep_dim,device=device)
    norm_data=pd.DataFrame()
    deep_SVDD.net_main.eval()
    c = [torch.zeros(deep_SVDD.net_main.rep_dim)]*cfg.settings['n_clusters']
    n_samples=[0]*cfg.settings['n_clusters']
    with torch.no_grad():
        train_loader, _ = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
        for data in train_loader:
            # get the inputs of the batch
            side,inputs, _ = data
            assignments = deep_SVDD.trainer.assignment
            
            inputs = inputs.to(device)
            outputs = deep_SVDD.net_main(inputs.float())
            
            logger.info(outputs[0])
            logger.info(outputs[1])
            
            for k in range(cfg.settings['n_clusters']):
                n_samples[k] = outputs[k].shape[0]
                
                c[k] = torch.transpose(torch.matmul(torch.transpose(outputs[k],0,1),assignments[k].float().view(outputs[k].shape[0],1)),0,1)[0]/assignments[k].sum()
    logger.info(c)
    for k in range(cfg.settings['n_clusters']):
        # c[k] /= n_samples[k]
        c[k][(abs(c[k]) < 0.01) & (c[k] < 0)] = -0.01
        c[k][(abs(c[k]) < 0.01) & (c[k] >= 0)] = 0.01
   
    #save center and weights
    with open(result+'/c.txt', 'w') as f:
        # f.write(','.join(str(i) for i in c))
        f.write(','.join(str(i) for i in deep_SVDD.c))
    for k in range(cfg.settings['n_clusters']):
        with open(result+'/c_'+str(k)+'.txt', 'w') as f:
            logger.info(c[k])
            f.write(','.join(str(i) for i in c[k].numpy()))
            # f.write(','.join(str(i) for i in deep_SVDD.c[k]))
        
    i=0
    for k in deep_SVDD.net_main.state_dict():
        with open(result+'/W_'+str(int(i/2))+'_'+str(i%2)+'.txt', 'w') as f:
            for item in deep_SVDD.net_main.state_dict()[k]:
                f.write(','.join(str(i) for i in item.numpy()))
                f.write('\n')
        i+=1
    np.set_printoptions(threshold=sys.maxsize)
    def scaler(df):
        x = df #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
      
        return x_scaled
    with torch.no_grad():
        _, test_loader = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
        for data in test_loader:
            side,inputs, _= data
            embeddings = torch.tensor(scaler(side.numpy())) #net_cond(side)
            # logger.info('embeddings: %s'%embeddings)
            test_cluster_assignments = deep_SVDD.trainer.cluster_responsibilities(embeddings,deep_SVDD.trainer.centroids, beta)
    test_cluster_assignments=torch.transpose(test_cluster_assignments,0,1)
    np.save(result+'/train_assignments', deep_SVDD.trainer.assignment.numpy())
    np.save(result+'/test_assignments', test_cluster_assignments.numpy())

    

if __name__ == '__main__':
    main()
