#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import Apps
from Apps import OurMNIST
from options import args_parser
import torch



if __name__=='__main__':
    ##########

    # 1. choose case
    args=args_parser()

    # 2. load datasets by apps
    # load data
    if args.case=='OurMNIST':
        FL_server=Apps.OurMNIST.OurMNIST(args)

    if args.simulation ==False:
        shapley_value=FL_server.train_FL()
    else:
        shapley_value=FL_server.train_FL_simu()

    for id in shapley_value:
        print(shapley_value[id])








