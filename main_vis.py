'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os
GPUS_EN = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_EN
import torch.backends.cudnn as cudnn
from baseline.vis.mainframe import MainFrame

from PyQt5.QtWidgets import QApplication
import sys

def main():
    cudnn.benchmark = True
    
    app = QApplication(sys.argv)
    ex = MainFrame(GPUS_EN)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
