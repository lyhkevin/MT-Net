def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']