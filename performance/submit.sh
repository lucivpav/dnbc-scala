BATCH_PATH='../batches/normal.csv'
TMP_BATCH='tmp.batch'
OUTPUTS_DIR='outputs'
cd $OUTPUTS_DIR
tail -n +2 $BATCH_PATH > $TMP_BATCH
while read p; do
	NWORKERS=`echo $p | cut -d' ' -f1`
	NCPUS=$(($NWORKERS+1))
	NAME=`echo "$p" | tr ' ' '-'`
	qsub -N "$NAME" -l select=1:ncpus=$NCPUS:mem=4gb:scratch_local=10gb:cluster=luna -l walltime=0:30:00 -v CONFIG="$p" ../job.sh
done < $TMP_BATCH
rm $TMP_BATCH
