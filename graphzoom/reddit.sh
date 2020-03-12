#!/bin/bash
for i in `seq 1 7`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 600;; #9
            (2)  echo 60;; #6
            (3)  echo 27;;
            (4)  echo 19;;
            (5)  echo 12;; #4
            (6)  echo 9;;
            (7)  echo 5;;
        esac)
        # python graphzoom.py -r ${ratio} -m deepwalk
#        python graphzoom.py -r ${ratio} -m deepwalk -pre /yushi/ -d reddit
        python graphzoom.py -r ${ratio} -m deepwalk -pre /mnt/yushi/ -d reddit 
done
