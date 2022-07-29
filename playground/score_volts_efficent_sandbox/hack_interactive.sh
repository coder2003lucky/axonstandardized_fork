#!/bin/bash



source ./input.txt

# for i in {1..25}; do
#     sh score_volts_efficent_sandbox/launch.sh $i &
# done
# wait


# for i in {25..50}; do
#     sh score_volts_efficent_sandbox/launch.sh $i &
# done
# # wait
# for i in {1..50}; do
#     sh score_volts_efficent_sandbox/launch.sh $i &
# done
# wait



# for i in {50..75}; do
#     sh score_volts_efficent_sandbox/launch.sh $i &
# done
# wait

for i in {1..38}; do
    sh score_volts_efficent_sandbox/launch.sh $i &
done


for i in {38..75}; do
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



















