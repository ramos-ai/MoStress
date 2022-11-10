#!/bin/bash
########################################################################################
#
#   @name       builder.sh
#   @author     Eng. William da Rosa Fröhlich
#   @date       09 11 2022
#   @desc       Script for MoStress environment configuration and installing requirements
#
########################################################################################

TAB="    "
PWD=$(pwd)
PREPROCESS_DIR="data/preprocessedData"

#Colors
RED='\033[0;31m'
NC='\033[0m'

printf "##################################################\n"
printf "#            MoSB - MoStress Builder             #\n"
printf "##################################################\n"

printf "\n$TAB Creating Paths\n"
for DIR in log $PREPROCESS_DIR/training $PREPROCESS_DIR/validation; do
    if [ ! -d "$PWD/$DIR" ]; then
        mkdir -p "$PWD/$DIR"
    fi
done 

printf "\n$TAB Installing Python Libraries\n"
pip install -r $PWD/requirements.txt -q

printf "\n${RED}##################################################${NC}"
printf "\n${RED}#        COPY YOUR DATASET TO 'data' PATH        #${NC}"
printf "\n${RED}##################################################${NC}"
printf "\n$TAB--> ${BLUE}$PWD/data\n\n${NC}"
