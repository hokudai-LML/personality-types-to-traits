import sys
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np


def spearman_corr(y_t, y_p):
    return spearmanr(y_p, y_t)[0]

def my_pearson(y_t, y_p):
  return pearsonr(y_t, y_p)[0]

def my_f1(y_t, y_p):
  return f1_score(y_true = y_t, y_pred = y_p, average = "macro")

def my_rmse(y_t, y_p):
  return np.sqrt(mean_squared_error(y_t, y_p))

variant = sys.argv[1]

if variant == "LR-N":
  table_rows = [("introverted",my_f1, "MBTI, I vs E (F1 score)"), 
              ("intuitive",my_f1, "MBTI, N vs P (F1 score)"),
              ("thinking",my_f1, "MBTI, T vs F (F1 score)"), 
              ("perceiving",my_f1, "MBTI, P vs J (F1 score)"), 
              ("enneagram_type",my_f1, "Enneagram (F1 score)"), 
              ("is_female",my_f1, "Gender (F1 score)"), 
              ("region",my_f1, "Region (F1 score)"), 
              ("agreeableness", my_pearson, "Big5 agreeableness (pearson) "),
              ("openness", my_pearson, "Big5 openness (pearson) "),
              ("conscientiousness", my_pearson, "Big5 conscientiousness (pearson) "),
              ("neuroticism", my_pearson, "Big5 neuroticism (pearson) "),
              ("extraversion", my_pearson, "Big5 extraversion (pearson) "),
              ("age", my_pearson, "Age (pearson) ")]
elif variant == "LR-NP":
  table_rows =[("agreeableness", my_pearson, "Big5 agreeableness (pearson) "),
              ("openness", my_pearson, "Big5 openness (pearson) "),
              ("conscientiousness", my_pearson, "Big5 conscientiousness (pearson) "),
              ("neuroticism", my_pearson, "Big5 neuroticism (pearson) "),
              ("extraversion", my_pearson, "Big5 extraversion (pearson) ")]
  # table_rows =[("agreeableness", spearman_corr, "Big5 agreeableness (spearman) "),
  #             ("openness", spearman_corr, "Big5 openness (spearman) "),
  #             ("conscientiousness", spearman_corr, "Big5 conscientiousness (spearman) "),
  #             ("neuroticism", spearman_corr, "Big5 neuroticism (spearman) "),
  #             ("extraversion", spearman_corr, "Big5 extraversion (spearman) ")]
else: 
  raise Exception("The summary script will not work properly for variants other than LR-N and LR-NP, which are created by the runallexperiments.sh")
file_object = open('results_output.txt', 'a')
for task_name, eval_metric, pretty_task_name in table_rows:
  df = pd.read_csv(os.path.join("./res/","-".join([task_name, variant]),"preds.csv"))
  scores = [eval_metric(y_t = df[df["fold"] == f]["true"], y_p = df[df["fold"] == f]["pred"]) for f in range(5)] 
  rmse = [my_rmse(y_t = df[df["fold"] == f]["true"], y_p = df[df["fold"] == f]["pred"]) for f in range(5)] 
  print("%s -- %.3f +- %.3f --- %.3f" % (pretty_task_name, np.mean(scores), np.std(scores), np.mean(rmse)))
  file_object.write("%s -- %.3f +- %.3f --- %.3f" % (pretty_task_name, np.mean(scores), np.std(scores), np.mean(rmse)))
  file_object.write("\n")

file_object.close()

