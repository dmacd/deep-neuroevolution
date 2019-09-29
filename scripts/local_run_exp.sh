#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
MODULE='es_distributed.main'
NUM_WORKERS=4
LOG_DIR='logs/'


MASTER_COMMAND=$(cat <<XXX
python -m ${MODULE} master \
        --master_socket_path /tmp/es_redis_master.sock \
        --algo ${ALGO} \
        --exp_file "${EXP_FILE}"
XXX
)

WORKER_COMMAND=$(cat <<XXX
python -m ${MODULE} workers \
        --master_host localhost \
        --relay_socket_path /tmp/es_redis_relay.sock \
        --algo ${ALGO} \
        --num_workers ${NUM_WORKERS}
XXX
)

tmux new -s $NAME -d
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME "${MASTER_COMMAND}" C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME "${WORKER_COMMAND}" C-m

tmux a -t $NAME
