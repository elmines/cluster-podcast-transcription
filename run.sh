#!/bin/bash

WHISPER_ROOT="/home/ethanlmines/blue_dir/repos/whisper.cpp/"
$WHISPER_ROOT/build/bin/whisper-cli --beam-size 1 -ocsv -np --model $WHISPER_ROOT/models/ggml-medium.bin sample.mp3 > /dev/null
# Writes sample.mp3.csv to working directory
