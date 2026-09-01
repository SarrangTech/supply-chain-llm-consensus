#!/bin/bash
# Usage: ./submit_shard.sh <framework> <small|large> <cost|bullwhip> <steps|full> <out_json>
set -e
FRAMEWORK=$1
TIER=$2
METRIC=$3
STEPS=$4
OUT=$5

if [ "$TIER" = "small" ]; then
  PART_LINES="#SBATCH --partition=short
#SBATCH --constraint=cascadelake
#SBATCH --cpus-per-task=56
#SBATCH --mem=32G
#SBATCH --exclusive"
else
  PART_LINES="#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G"
fi

if [ "$STEPS" = "full" ]; then
  STEPS_ARG=""
  TIME="08:00:00"
else
  STEPS_ARG="--steps ${STEPS}"
  TIME="01:00:00"
fi

SCRIPT=$(mktemp /tmp/shard_XXXXXX.sh)
cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${FRAMEWORK:0:8}-${TIER:0:2}-${METRIC:0:4}
${PART_LINES}
#SBATCH --time=${TIME}
#SBATCH --output=/home/bhawesh/supply-chain-llm-consensus/logs/shard_${FRAMEWORK}_${TIER}_${METRIC}_%j.log

set -x
# Unique port per job, so co-resident jobs on a shared GPU node do not
# collide on the default 11434 (see NOTES_AND_ASSUMPTIONS.md section (f)).
PORT=\$((20000 + SLURM_JOB_ID % 10000))
export OLLAMA_HOST=127.0.0.1:\${PORT}
export OLLAMA_BASE_URL=http://127.0.0.1:\${PORT}
export no_proxy="localhost,127.0.0.1,\${no_proxy}"
export NO_PROXY="localhost,127.0.0.1,\${NO_PROXY}"
export OLLAMA_MODELS=/home/bhawesh/ollama/models
/home/bhawesh/ollama/bin/ollama serve &
SERVER_PID=\$!
for i in \$(seq 1 30); do curl -s http://127.0.0.1:\${PORT} > /dev/null && break; sleep 1; done

cd /home/bhawesh/supply-chain-llm-consensus
module load python/3.13.5
source venv/bin/activate
time python run_experiments.py ${STEPS_ARG} \
  --only-framework ${FRAMEWORK} --only-model-tier ${TIER} --only-metric ${METRIC} \
  --out ${OUT}

kill \$SERVER_PID 2>/dev/null || true
EOF
sbatch --parsable "$SCRIPT"
