seed_arr=(1011) # seed
minigame_arr=('27m_vs_30m') # scenario
algo_arr=('RNN_AGENT/qmix_ssl_beta') # ('RNN_AGENT/qmix_ssl_beta' 'RNN_AGENT/qmix_beta')
masking_ratio=(0.2 0.4 0.6 0.8)

for seed in ${seed_arr[@]}
do
for minigame in ${minigame_arr[@]} 
do
for ratio in ${masking_ratio[@]}
do
for algo in ${algo_arr[@]}
do
    python run_smac.py --seed $seed --minimap $minigame --masking-ratio $ratio --algorithm $algo
done
done
done
done