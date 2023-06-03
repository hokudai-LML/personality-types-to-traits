# NGRAM 
python3 baselines.py  -data_path "./data" -label allmbti -tasktype classification -folds mbti -feats 1gram -model lr -variant LR-N
python3 baselines.py  -data_path "./data" -label allbig5 -tasktype regression -folds big5 -feats  1gram -model lr -variant LR-N
python3 baselines.py  -data_path "./data" -label enneagram_type -tasktype classification -folds enneagram -feats 1gram -model lr -variant LR-N

python3 baselines.py  -data_path "./data" -label age -tasktype regression -folds age -feats 1gram -model lr -variant LR-N
python3 baselines.py  -data_path "./data" -label is_female -tasktype classification -folds gender -feats 1gram -model lr -variant LR-N
python3 baselines.py  -data_path "./data" -label region -tasktype classification -folds region -feats 1gram -model lr -variant LR-N

# BIG5 -- NGRAM + PREDS
python3 baselines.py  -data_path "./data" -label allbig5 -tasktype regression -folds big5 -feats 1gram,mbtipred,ennepred -model lr -variant LR-NP

python3 summarize_res.py LR-NP