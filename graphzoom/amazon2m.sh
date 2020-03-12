#!/bin/bash
for i in `seq 7 10`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 12;; #4
            (2)  echo 5;;
            (3)  echo 9;;
            (4)  echo 90;;
            (5)  echo 60;;
            (6)  echo 27;; # 5
            (7)  echo 12;;
            (8)  echo 9;;
            (9)  echo 5;;
            (10)  echo 2;;
        esac)
        python graphzoom.py -r ${ratio} -m deepwalk -d Amazon2M -pre  /yushi 
done
