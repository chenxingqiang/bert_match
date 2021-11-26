#!/usr/bin/env bash
# test split user
for i in $(seq -w 0 133); do cat 20210419-refine-0-part-${i}.txt| grep  "^73537 " | wc -l ;done
