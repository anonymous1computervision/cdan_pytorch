import torch
import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms
from utils.log import get_machinelearning_logger, cdan_train_logger
import dixitool.utils as util
from dixitool.data.datasetsFactory import create_data_loader_with_transform
import torch.optim as optim
import train_test.office31_train_test as office_transfer
import os
from loss.loss import cdan_domain_loss
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

def train(base_network, adversal_net, optimizer, src_trainloader, tgt_trainloader, config, epoch,n_epoch,device=None,closure=None):
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


        if config['method']  == 'CDAN':
            transfer_loss = cdan_domain_loss(features, softmax_out, src_inputs.size(0), tgt_inputs.size(0), adversal_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, src_labels)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()

        if closure is not None:
            closure(batch_idx, num_iter, epoch, n_epoch, classifier_loss, transfer_loss)





def test():
    print()



def main():
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--source', type=str, default='dslr', help="The source dataset")
    parser.add_argument('--target', type=str, default='webcam', help="The target dataset")
    parser.add_argument('--src_path', type=str, default='../data/domain_adaptation_images/dslr', help="The source dataset path")
    parser.add_argument('--tgt_path', type=str, default='../data/domain_adaptation_images/webcam', help="The target dataset path")
    parser.add_argument('--gpu_id', type=str, nargs='?', default=None, help="device id to run")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--n_epoch',type=int,default=100,help="number of epochs")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train_transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
    ])
    #dataset 没有transform会报错：TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.Image.Image'>
    src_trainloader = create_data_loader_with_transform(args.source , args.src_path,train_transform=train_transform , batch_size=36)
    tgt_trainloader = create_data_loader_with_transform(args.target , args.tgt_path,train_transform=train_transform , batch_size=36)
    tgt_test_loader = tgt_trainloader
    #network
    #office 31 classes
    #
    backbone = ResNetFc("ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=31)
    # backbone的output_num()，表示传给self.fc的数据的维度大小
    ad_net =  AdversarialNetwork(backbone.output_num()*31, 1024)

    #返回的参数中有“lr_mult":10, 'decay_mult':2
    parameter_list = backbone.get_parameters() + ad_net.get_parameters()
    #optimizer
    optimizer = optim.SGD( parameter_list, lr=args.lr, momentum=0.9, weight_decay=0.0005,nesterov=True)
    
    #训练设置
    config={}
    
    config["loss"] = {"trade_off":1.0}
    config["method"] = args.method

    #logger
    train_logger = get_machinelearning_logger(cdan_train_logger)
    #source data and target data
    for epoch in range(args.n_epoch):
        print('Epoch: {}/{}'.format(epoch,args.n_epoch))
        train(backbone, ad_net, optimizer, src_trainloader,tgt_trainloader, config, epoch, args.n_epoch,device=base_device,closure=train_logger)
        test()




if __name__ == '__main__':    
    main()






