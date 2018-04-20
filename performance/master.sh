#!/bin/bash
trap 'clean_scratch' TERM EXIT
cd $SCRATCHDIR || exit 1

module add jdk-8

TMP_DIR="/dev/shm/tmp"
mkdir $TMP_DIR
JAVA_OPTS="-Djava.io.tmpdir=$TMP_DIR"

SPARK_DAEMON_JAVA_OPTS="$JAVA_OPTS" /storage/praha1/home/lucivpav/spark-2.1.0-bin-hadoop2.7/sbin/start-master.sh

sleep infinity
