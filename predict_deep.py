import os
import time
import argparse
import torch.autograd
from skimage import io
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import RS_LEVIR as RS
from models.model_sets.snunet import SNUNet as Net


class PredOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default=r'D:\dl_file\DL_FILE\LEVIA-CD\shiyan_256\test', help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default=r'D:\dl_file\DL_FILE\LEVIA-CD\shiyan_256\Pre\SNUNET',
                            help='directory to output masks. me: if use /PRED_DIR/, then will create result in ')
        # 读取保存的模型文件的路径
        parser.add_argument('--chkpt_path', required=False,
                            default=working_path + '/checkpoints/',
                            help="me: may be here need to choice own train pth file, so I rename xxx.pth to now name")
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def predict(net, pred_set, pred_loader, pred_dir):
    pred_dir_rgb = os.path.join(pred_dir, 'change_rgb')

    if not os.path.exists(pred_dir_rgb):
        os.makedirs(pred_dir_rgb)

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()

        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad():
            # out_change, Sgb1 = net(imgs_A, imgs_B)
            out_change = net(imgs_A, imgs_B)
            out_change = F.sigmoid(out_change[0])

        # 变化区域
        change_mask = out_change.cpu().detach() < 0.5
        change_mask = change_mask.squeeze()

        pred_path = os.path.join(pred_dir_rgb, mask_name)

        io.imsave(pred_path, RS.Index2Color(change_mask))

def main():
    begin_time = time.time()
    preoptions = PredOptions()
    opt = preoptions.parse()
    net = Net().cuda()
    net.load_state_dict(torch.load(opt.chkpt_path), strict=False)
    net.eval()

    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


if __name__ == '__main__':
    main()
