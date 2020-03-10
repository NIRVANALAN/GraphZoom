#!/bin/bash
for i in `seq 1 5`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 60;;
            (2)  echo 27;;
            (3)  echo 19;;
            (4)  echo 9;;
            (5)  echo 5;;
        esac)
        # python graphzoom.py -r ${ratio} -m deepwalk
        python graphzoom.py -r ${ratio} -m deepwalk -pre /yushi/ -d reddit
        # python graphzoom.py -r ${ratio} -m deepwalk -pre /mnt/yushi/ -d reddit
done
