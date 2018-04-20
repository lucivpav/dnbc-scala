MASTER_URL="spark://luna11.fzu.cz:7077"
NSLAVES=8
CPUS_PER_SLAVE=2
RAM_PER_SLAVE=8

HOMEDIR="/storage/praha1/home/lucivpav"
OUTPUT_DIR="${HOMEDIR}/outputs/slaves"

mkdir -p $OUTPUT_DIR
if [[ $? -ne 0 ]] ; then
	exit 1
fi

cd $OUTPUT_DIR

for i in $(seq 1 $NSLAVES)
do
	qsub -N "slave-${i}" -l select=1:ncpus=$CPUS_PER_SLAVE:mem=15gb:scratch_local=10gb:cluster=luna -l walltime=1:00:00 -v MASTER_URL=${MASTER_URL},CPUS=${CPUS_PER_SLAVE},RAM=${RAM_PER_SLAVE} $HOMEDIR/slave.sh
done
