#!/bin/bash

NAME='s5_train.py'
ps -ef | grep "$NAME"
echo "---------------"
echo $NAME
ID=`ps -ef | grep "$NAME" | grep -v "grep" | awk '{print $2}'`
echo $ID
for id in $ID
do
kill -9 $id
echo "killed $id"
done
echo "---------------"
