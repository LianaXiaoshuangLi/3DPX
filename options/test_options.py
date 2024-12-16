from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        # self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        # self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # self.parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--txt_file', type=str, default='', help='txt file for test')
        # self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--subset', type=str, default='', help='txt file for test')
        self.parser.add_argument('--dim', type=int)
        self.parser.add_argument('--joint', default=False, action='store_true')
        self.parser.add_argument('--three', default=False, action='store_true')
        self.parser.add_argument('--recons', type=str)
        
        self.isTrain = False
