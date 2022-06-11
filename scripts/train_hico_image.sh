#!/bin/bash

CFG=$1
OPTS=${@:2}

python train_hico.py --config-file $CFG $OPTS 2>&1
