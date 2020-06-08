#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

# TODO: Create MODEL variable
MODEL=$1
# TODO: Create DEVICE variable
DEVICE=$2
# TODO: Create VIDEO variable
VIDEO=$3
QUEUE=$4
OUTPUT=$5
# TODO: Create PEOPLE variable
PEOPLE=$6

mkdir -p $5

if echo "$DEVICE" | grep -q "FPGA"; then # if device passed in is FPGA, load bitstream to program FPGA
    #Environment variables and compilation for edge compute nodes with FPGAs
    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2

    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-2_PL2_FP16_MobileNet_Clamp.aocx

    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

python3 person_detect.py  --model ${MODEL} \
                          --device ${DEVICE} \
                          --video ${VIDEO} \
                          --queue_param ${QUEUE} \
                          --output_path ${OUTPUT}\
                          --max_people ${PEOPLE} \

cd /output

tar zcvf output.tgz *