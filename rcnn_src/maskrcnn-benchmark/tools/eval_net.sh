#!/bin/bash

CONFIG_FILE="configs/xv_frcnn_R50_FPN.yaml"
MIN_ITER=0

for filename in "$@"
do
    echo "$filename"
    [ -e "$filename" ] || continue
    python tools/eval_net.py --config-file "$CONFIG_FILE" \
                             --min-iter "$MIN_ITER" \
                             --checkpoint-file "$filename"
    retval=$?
    if [ $retval -ne 0 ]; then
        echo "exiting"
        break
    fi
done
