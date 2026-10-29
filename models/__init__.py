
def getModel(opt):
    name = opt['model']
    if 'dinov3IR_abdomen' in name:
        from .dinov3IR_abdomen import Dinov3IR
        model = Dinov3IR(**opt['nkwargs'])
    elif 'dinov3IR_acdc' in name:
        from .dinov3IR_acdc import Dinov3IR
        model = Dinov3IR(**opt['nkwargs'])
    else:
        raise ValueError('Unsupported release model: %s.' % name)
    model = model.cuda()
    print('----->>>> Model %s is built ...' % name)
    return model
