MASTER_URL="spark://luna12.fzu.cz:7077"
NSLAVES=2
CPUS_PER_SLAVE=1
RAM_PER_SLAVE=8

for i in $(seq 1 $NSLAVES)
do
	qsub -N "slave-${i}" -l select=1:ncpus=$CPUS_PER_SLAVE:mem=15gb:scratch_local=10gb:cluster=luna -l walltime=0:30:00 -v MASTER_URL=${MASTER_URL},CPUS=${CPUS_PER_SLAVE},RAM=${RAM_PER_SLAVE} slave.sh
done
