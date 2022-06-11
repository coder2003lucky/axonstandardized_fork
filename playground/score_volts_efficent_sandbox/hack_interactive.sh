#!/bin/bash



source ./input.txt

for i in {1..35}; do
    sh score_volts_efficent_sandbox/launch.sh $i &
done
wait


for i in {35..70}; do
    sh score_volts_efficent_sandbox/launch.sh $i &
done
wait


# for i in {1..30}; do
#     sh score_volts_efficent_sandbox/launch.sh $i &
# done
# wait


# for i in {30..55}; do
#     sh score_volts_efficent_sandbox/launch.sh $i &
# done
# wait
# for i in {55..80}; do
#     sh score_volts_efficent_sandbox/launch.sh $i &
# done
# wait



















