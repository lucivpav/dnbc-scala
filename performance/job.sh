#!/bin/bash
HOMEDIR="/storage/praha1/home/lucivpav"
trap 'clean_scratch' TERM EXIT
cd $SCRATCHDIR || exit 1

TMP_DIR="/dev/shm/tmp"
mkdir $TMP_DIR

module add jdk-8
RESULTS_FILE="$CONFIG.results"
JAVA_OPTS="-Xmx8G -Djava.io.tmpdir=$TMP_DIR" $HOMEDIR/pack/bin/main $CONFIG > "$RESULTS_FILE"
cp "$RESULTS_FILE" $DATADIR || export CLEAN_SCRATCH=false
