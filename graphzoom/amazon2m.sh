#!/bin/bash
for i in `seq 4 8`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 12;; #4
            (2)  echo 5;;
            (3)  echo 9;;
            (4)  echo 90;;
            (5)  echo 60;;
            (6)  echo 27;;
            (7)  echo 19;;
            (8)  echo 9;;
        esac)
        python graphzoom.py -r ${ratio} -m deepwalk -d Amazon2M -pre  /yushi
done
