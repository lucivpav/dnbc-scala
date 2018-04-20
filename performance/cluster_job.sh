#!/bin/bash

MASTER_URL="spark://luna11.fzu.cz:7077"

trap 'clean_scratch' TERM EXIT
cd $SCRATCHDIR || exit 1

module add jdk-8
RESULTS_FILE="$CONFIG.results"

TMP_DIR="/dev/shm/tmp"
mkdir $TMP_DIR

LIB_DIR="/storage/praha1/home/lucivpav/pack/lib"
CORE="$LIB_DIR/core_2.11-0.1.0-SNAPSHOT.jar"
TEST_UTILS="$LIB_DIR/testutils_2.11-0.1.0-SNAPSHOT.jar"
PERFORMANCE="$LIB_DIR/performance_2.11-0.1.0-SNAPSHOT.jar"

JARS="--jars ${CORE},${TEST_UTILS}"
DRIVER_PATH="--driver-class-path ${CORE}:${PERFORMANCE}:${TEST_UTILS}"
JAVA_OPTS="-Djava.io.tmpdir=$TMP_DIR"

/storage/praha1/home/lucivpav/spark-2.1.0-bin-hadoop2.7/bin/spark-submit --conf "spark.driver.extraJavaOptions=$JAVA_OPTS" --conf "spark.executor.extraJavaOptions=$JAVA_OPTS" --executor-memory 8G --driver-memory 8G --name "Simple App" --deploy-mode cluster --master $MASTER_URL $DRIVER_PATH $JARS --class Main $PERFORMANCE $CONFIG > "$RESULTS_FILE" 2>&1

cp "$RESULTS_FILE" $DATADIR || export CLEAN_SCRATCH=false
