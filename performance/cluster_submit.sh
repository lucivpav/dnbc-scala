CPUS_PER_SLAVE="2"
BATCH_NAME='cluster'
HOMEDIR="/storage/praha1/home/lucivpav"
BATCH_PATH="${HOMEDIR}/batches/${BATCH_NAME}.csv"
TMP_BATCH='tmp.batch'
ID="${BATCH_NAME}_${CPUS_PER_SLAVE}_${RANDOM}"
OUTPUT_DIR="${HOMEDIR}/outputs/$ID"
LOGDIR="${HOMEDIR}/log/$ID"

mkdir $OUTPUT_DIR $LOGDIR
if [[ $? -ne 0 ]] ; then
	exit 1
fi

cd $OUTPUT_DIR
tail -n +2 $BATCH_PATH > $TMP_BATCH
while read p; do
	NAME=`echo "$p" | tr ' ' '-'`
	qsub -N "${NAME}_${ID}" -l select=1:ncpus=1:mem=15gb:scratch_local=10gb:cluster=luna -l walltime=0:30:00 -v CONFIG="$p",DATADIR="$LOGDIR" $HOMEDIR/cluster_job.sh
done < $TMP_BATCH
rm $TMP_BATCH
