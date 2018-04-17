#!/bin/bash
trap 'clean_scratch' TERM EXIT
cd $SCRATCHDIR || exit 1

module add jdk-8

/storage/praha1/home/lucivpav/spark-2.1.0-bin-hadoop2.7/sbin/start-slave.sh $MASTER_URL -m ${RAM}G -c ${CPUS}

sleep infinity
