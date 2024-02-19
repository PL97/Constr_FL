#/usr/env bash

for client_n in 1 5 10 20; do
    for random_idx in {1..10}; do
        sbatch fairClassification.slurm $client_n $random_idx
    done
done