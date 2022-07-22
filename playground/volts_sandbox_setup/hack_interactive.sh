#!/bin/bash



source ./input.txt

for i in {1..25}; do
    sh volts_sandbox_setup/launch.sh $i &
done
wait

for i in {25..50}; do
    sh volts_sandbox_setup/launch.sh $i &
done
wait


for i in {50..75}; do
    sh volts_sandbox_setup/launch.sh $i &
done
wait
# for i in {60..80}; do
#     sh volts_sandbox_setup/launch.sh $i &
# done




















