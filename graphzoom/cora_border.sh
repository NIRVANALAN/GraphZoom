#!/bin/bash
for i in `seq 1 1`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 2;;
            (2)  echo 2;;
#            (3)  echo 9;;
#            (4)  echo 19;;
        esac)
        python graphzoom.py -r ${ratio} -m deepwalk -d cora -pj border
done
