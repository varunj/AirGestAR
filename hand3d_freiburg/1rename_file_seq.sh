#!/bin/bash
num=1
for file in *.mp4; do
       mv "$file" "$(printf "train_bloom_%u" $num).mp4"
       let num=$num+1
done