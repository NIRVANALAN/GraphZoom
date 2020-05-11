#!/bin/bash
for i in `seq 10 10`    ## coarsening level
    do
        ratio=$(case "$i" in
            # (1)  echo 600;; #9
            # (2)  echo 400;; #9
            # (3)  echo 300;; #9
            # (4)  echo 200;; #9
            # (5)  echo 100;; #9
            # (6)  echo 60;; #6
            # (7)  echo 27;; #5
            # (8)  echo 12;; #4
            # (9)  echo 5;; #3
            (10)  echo 2;; 
            (11)  echo 1;; 
        esac)
        # python graphzoom.py -r ${ratio} -m deepwalk
#        python graphzoom.py -r ${ratio} -m deepwalk -pre /yushi/ -d reddit
#        python graphzoom.py -r ${ratio} -m deepwalk -pre /mnt/yushi/ -d reddit 
        #python graphzoom.py -r ${ratio} -m ft -pre /mnt/yushi/ -d reddit --proj border
        python graphzoom.py -r ${ratio} -m deepwalk -pre /data/data0/yushi -d reddit --proj border
done
