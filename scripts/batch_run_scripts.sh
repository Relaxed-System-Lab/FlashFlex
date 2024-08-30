model_size="llama-7b"

rank=0
save_logs=$1
pkill python
bash scripts/generate_hetero_scripts.sh


for script in ./llama-scripts-logs/${model_size}-scripts/*.sh; do
    echo "Running "$script
    if [[ $save_logs == "save" ]];
    then
        bash $script > ./llama-scripts-logs/${model_size}-scripts/${rank}.txt & 
        rank=$((1 + $rank))
    else
        bash $script & 
    fi
done
