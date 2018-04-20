MASTER_CPUS=4
MASTER_RAM=15

HOMEDIR="/storage/praha1/home/lucivpav"
OUTPUT_DIR="${HOMEDIR}/outputs/master"

mkdir -p $OUTPUT_DIR
if [[ $? -ne 0 ]] ; then
	exit 1
fi

cd $OUTPUT_DIR
qsub -N master -l select=1:ncpus=$MASTER_CPUS:mem=${MASTER_RAM}gb:scratch_local=10gb:cluster=luna -l walltime=1:00:00 $HOMEDIR/master.sh
