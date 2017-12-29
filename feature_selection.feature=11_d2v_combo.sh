#!/usr/bin/env bash

# 0.all, [1-8] each feature set, [9-13] combined features
# do feature selection on 9 and 11
export FEATURE_ID=11

export CONTEXT=0
declare -a power=(4 5 6 7 8 9 10 11 12)

for NUMBER_TO_KEEP in "${power[@]}"
do
    export NUMBER_TO_KEEP;
    echo "Feature Selection, Feature Number=$NUMBER_TO_KEEP, Context=$CONTEXT, Feature=$FEATURE_ID, with similarity";
done