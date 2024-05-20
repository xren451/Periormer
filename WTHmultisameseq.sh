### M Different Seq_length and different prediction length

python3 -u main_periormer.py --model periormer --root_path './data/Environment' --data_path 'WTH.csv' --data 'WTH'   --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 --per_term 25
 
python3 -u main_periormer.py --model periormer --root_path './data/Environment' --data_path 'WTH.csv' --data 'WTH'  --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 --per_term 25

python3 -u main_periormer.py --model periormer --root_path './data/Environment' --data_path 'WTH.csv' --data 'WTH'   --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 --per_term 25

python3 -u main_periormer.py --model periormer --root_path './data/Environment' --data_path 'WTH.csv' --data 'WTH'  --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 --per_term 25

python3 -u main_periormer.py --model periormer --root_path './data/Environment' --data_path 'WTH.csv' --data 'WTH'  --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 --per_term 25
