import torch
import torch.nn as nn
label_fromsrc = 1.0
label_fromtgt = 0.0
internal_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def cdan_domain_loss(feature, softmax_prediction, src_batch_size, tgt_batch_size, adversarial_net):
    # type: (tensor,tensor,int,int, nn.Module) -> scalar tensor
    """calculate cdan entropy loss

    Args:
        feature(tensor): 2-D shape tensor, inputs of classifier layer . torch.cat((sourcefeature, targetfeature), dim=0)
        softmax_prediction(tensor): the output of classfier（detached）. comes from a nn.Linear Module. and goes through a nn.Softmax(Not nn.LogSoftmax) module
        adversarial_net(nn.Module): adversarial network
    Return:
        cdan loss
    """
    # multi-linear mapping
    multi_linear = torch.bmm(softmax_prediction.unsqueeze(2), feature.unsqueeze(1))
    #adversarial output should make to 1-D shape, in order to have the same shape as labels
    adversarial_out = adversarial_net(multi_linear.view(-1, softmax_prediction.size(1)*feature.size(1))).view(-1)
    #the shape of 2d-shape-tensor adversarial_out is Size([ feature.size(0),1])
    src_domain_label = torch.full((src_batch_size,), label_fromsrc ,device=internal_device)
    tgt_domain_label = torch.full((tgt_batch_size,), label_fromtgt ,device=internal_device)

    domain_label = torch.cat((src_domain_label,tgt_domain_label),dim=0)

    return nn.BCELoss()(adversarial_out, domain_label)


def cdan_e_loss():
    raise NotImplementedError("NotImplementedError")
