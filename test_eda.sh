seed_arr=(22022) # seed
minigame_arr=('27m_vs_30m') # scenario
algo_arr=('RNN_AGENT/qmix_ssl_beta') # ('RNN_AGENT/qmix_ssl_beta' 'RNN_AGENT/qmix_beta')
momentum_ratio=(0.9 0.99 0.999 0.9999)

for seed in ${seed_arr[@]}
do
for minigame in ${minigame_arr[@]} 
do
for algo in ${algo_arr[@]}
do
for ratio in ${momentum_ratio[@]}
do
    python run_smac.py --seed $seed --minimap $minigame --momentum $ratio --algorithm $algo
done
done
done
done