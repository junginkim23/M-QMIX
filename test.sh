seed_arr=(11 22 33 44 55 66) # seed 
minigame_arr=('5m_vs_6m' '2c_vs_64zg' '2s3z' '2s_vs_1sc') # scenario
algo_arr=('RNN_AGENT/qmix_beta' 'RNN_AGENT/qmix_ssl_beta') 

for seed in ${seed_arr[@]}
do
for minigame in ${minigame_arr[@]} 
do
for algo in ${algo_arr[@]}
do
    python run_smac.py --seed $seed --minimap $minigame --algorithm $algo
done
done
done