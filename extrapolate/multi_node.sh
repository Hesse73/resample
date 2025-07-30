num_nodes=${1:-1}
total_size=${2:-32}
# 如果 num_nodes 大于 1, 根据 $KUBERNETES_POD_NAME 的后缀设置 seed
# 如果 KUBERNETES_POD_REPLICA_TYPE == Master, 则seed=0
# 否则 seed=$KUBERNETES_POD_NAME 的后缀id (格式为 xxx-worker-id)
seed=0
if [ "$num_nodes" -gt 1 ]; then
  if [ "$KUBERNETES_POD_REPLICA_TYPE" == "Worker" ]; then
    seed=${KUBERNETES_POD_NAME##*-}
    seed=$((seed + 1))  # 因为 seed 从 0 开始，所以加 1
  fi
fi
split_size=$((total_size / num_nodes))
echo "num_nodes: $num_nodes, total_size: $total_size, split_size: $split_size"
echo "Current seed: $seed"

shift 2
echo "Running command: --seed $seed --n $split_size $@"
python extrapolate.py --seed $seed --n $split_size $@
