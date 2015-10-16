#!/usr/bin/env python
# encoding: utf-8
"""
sheller_great3.py

Created by Michael Schneider on 2015-10-16
"""

import argparse
import sys
import os.path
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/sheller_great3.py.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description='My program description.')
    parser.add_argument('outdir', help='Output directory')

    args = parser.parse_args()
    logging.debug('Program started')

    outdir = os.path.abspath(args.outdir)

    logging.debug('Program finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
