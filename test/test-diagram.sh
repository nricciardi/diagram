#!/bin/bash

python3 $1 \
--classifier classifier.pth \
--bbox-detector bbox_detector.pth \
--input easy-flow.png \
--outputs-dir-path ./outcome
# --then-compile

