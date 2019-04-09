

def get_machinelearning_logger(func):
    def logger(*argv):
        func(*argv)
    return logger

#logger for dann domain adaption algorithm in one epoch
def cdan_train_logger(batch_idx, len_batch, epoch, n_epoch, classifier_loss, transfer_loss):
    if batch_idx % 5 == 0 or batch_idx == (len_batch-1):
        #print loss
        print('[batch:{}/{}]\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(batch_idx+1,len_batch,classifier_loss, transfer_loss ) )

def cdan_test_logger(epoch, n_epoch,src_corrects, src_data_sum, tgt_corrects, tgt_data_sum,domain_corrects):
    print('source acc: {:.4f}%\ttarget acc:{:.4f}%\tdomain acc:{:.4f}%'.format( 100. * float(src_corrects) / src_data_sum, 
        100. * float(tgt_corrects)/tgt_data_sum, 100. * float(domain_corrects)/(src_data_sum+tgt_data_sum)))