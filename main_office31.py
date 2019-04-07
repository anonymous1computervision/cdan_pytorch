import torch
import argparse
import torch.nn as nn
import torchvision
import dixitool.utils as util
import torch.optim as optim
import train_test.office31_train_test as office_transfer
import os
import loss
from models.network import ResNetFc, AdversarialNetwork

base_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#script can chose source and target images from amazon, webcam and dslr 

#learning rate scheduler
def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer
#use Resnet50

    #a->d, d->w 是0.0003 其他的迁移是0.001
schedule_param={"lr":0.001, "gamma":0.001, "power":0.75}

def train(base_network, adversal_net, optimizer, src_trainloader, tgt_trainloader, config, epoch,n_epoch,device=None):
    len_src = len(src_trainloader)
    len_tgt = len(tgt_trainloader)
    num_iter = max(len_src, len_tgt)
    #########
    #Train
    #########
    base_network.train()
    adversal_net.train()
    
    for batch_idx in range(num_iter):

        i = epoch*num_iter+batch_idx
        # init optimizer
        optimizer = inv_lr_scheduler(optimizer, i,**schedule_param )
        optimizer.zero_grad()

        if batch_idx % len_src == 0:
            iter_src = iter(src_trainloader)
        if batch_idx % len_tgt == 0:
            iter_tgt = iter(tgt_trainloader)

        src_inputs, src_labels = iter_src.next()
        tgt_inputs, tgt_labels = iter_tgt.next()
        # to cuda variable
        if device is not None:
            src_inputs ,src_labels = src_inputs.to(device), src_labels.to(device)
            tgt_inputs = tgt_inputs.to(device)

        #调整learning rate
        """
            code...
        """

        features_source, outputs_source = base_network(src_inputs)
        features_target, outputs_target = base_network(tgt_inputs)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        loss_params = config["loss"]


        if config['method'] == 'CDAN+E':           
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], adversal_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], adversal_net, None, None, random_layer)
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, adversal_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, src_labels)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()






def test():
    print()



def main():
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--source', type=str, default='dslr', help="The source dataset")
    parser.add_argument('--target', type=str, default='webcam', help="The target dataset")
    parser.add_argument('--gpu_id', type=str, nargs='?', default=None, help="device id to run")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    #network
    #office 31 classes
    #
    backbone = ResNetFc("Resnet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=31)
    # backbone的output_num()，表示传给self.fc的数据的维度大小
    ad_net =  AdversarialNetwork(backbone.output_num()*31, 1024)

    #返回的参数中有“lr_mult":10, 'decay_mult':2
    parameter_list = backbone.get_parameters() + ad_net.get_parameters()
    #optimizer
    optimizer = optim.SGD( parameter_list, lr=args.lr, momentum=0.9, weight_decay=0.0005,nesterov=True)
    
    #训练设置
    config={}
    
    config["loss"] = {"trade_off":1.0}

    #source data and target data
    for epoch in range(n_epoch):
        office_transfer.train(backbone, ad_net, optimizer, src_trainloader,tgt_trainloader, config, epoch, n_epoch,device=base_device)
        office_transfer.test()




if __name__ == '__main__':    
    main():







