#!/bin/bash
HOMEDIR="/storage/praha1/home/lucivpav"
DATADIR="$HOMEDIR/log/"
trap 'clean_scratch' TERM EXIT
cd $SCRATCHDIR || exit 1

module add jdk-8
RESULTS_FILE="$CONFIG.results"
JAVA_OPTS=-Xmx3G $HOMEDIR/pack/bin/main $CONFIG > "$RESULTS_FILE"
cp "$RESULTS_FILE" $DATADIR || export CLEAN_SCRATCH=false
