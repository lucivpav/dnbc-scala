MASTER_CPUS=4
MASTER_RAM=15

qsub -N master -l select=1:ncpus=$MASTER_CPUS:mem=${MASTER_RAM}gb:scratch_local=10gb:cluster=luna -l walltime=0:30:00 master.sh
