import pandas as pd
import numpy as np
import sys
import argparse
import math
import os
from random import randint
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, f_regression, f_classif, VarianceThreshold
from sklearn.metrics import f1_score, r2_score, precision_score, recall_score
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, HuberRegressor
import pickle
from scipy.stats import pearsonr 
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import argparse
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import itertools
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
import json
from features import features_dict
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import RANSACRegressor

def pearson_scorer(estimator, X, y):
       return pearsonr(estimator.predict(X), y)[0]


def calc_rmse(x,y):
  return(-math.sqrt(mean_squared_error(x,y)))

def pearson_corr(x,y):
       return pearsonr(x,y)[0]   


reg_eval_func = pearson_corr  # or calc_rmse


def load_data(unames, texts, labels_path, label_name, label_type, fold_grp, repeat, args): # percentiles or scores THAT PARAMETER IS DEPRECATED DOES NOTHING NOW
  print("Loading metadata ...")
  # load labels
  ldf = pd.read_csv(labels_path)

  auth2label = dict(zip(list(ldf["author"]), list(ldf[label_name])))
  labels = [auth2label[x] for x in unames]

  # generate folds (or load from disk where available)
  folds_filename = os.path.join(args.data_path, "folds.csv")
  
  print("Loading the folds file ...")
  if os.path.isfile(folds_filename):
    fdf = pd.read_csv(folds_filename)

    fdf = fdf[fdf["group"] == fold_grp]
    fdf = fdf[fdf["trait"] == label_name]
    fdf = fdf[fdf["repeat"] == repeat]

    auth2fold = dict(zip(list(fdf["author"]), list(fdf["fold"])))
    folds = []
    nonfails, fails = 0, 0
    for x in unames:
      if x in auth2fold:
        nonfails += 1
        folds.append(auth2fold[x])
      else:
        fails += 1
        folds.append(None)
    assert len(folds) == len(unames) # this also has to be true
    assert len(folds) == len(labels)
    if len(auth2fold) != nonfails:# if this is true then there are no examples that exist in the folds list but not in the data
      print("Warning, there are some users in the folds file that are not in the data (maybe you removed mbti subreddits from the data but not from the folds), these users will be ignored!")

    for i in range(len(folds)):
        if folds[i] is None:
                labels[i] = math.nan # later, everything where the label is None is thrown out, including these users for which we have no fold indexes
  else:
        raise Exception("Fold file not found.")
  ret_df = pd.DataFrame(list(zip(unames, folds, labels)), columns =['author', 'fold','label'])
  return(ret_df)




# precompute n-gram counts because i think the tfidf later is fast and i want to try different weighings without gathering the counts again every time)
def precompute_or_load_feats(data, feats_id_prefix, args):
  # *** all *** means all except n-gram feats
  all_feats_names = [] 
  all_feats_list = []
  
  text_feats_names = []
  text_feats_matrix = None
  grams_finished = False
  slen_finished = False
  argssplitlist = [x.strip() for x in args.feats.split(",")]
  PATH_PREDICTION_FEATS = os.path.join(args.data_path, "mbti_enne_pred_extended_liwc.csv")

  for feats_type in argssplitlist:
    if "gram" in feats_type and not grams_finished:

      feats_filename = os.path.join(feats_id_prefix, "feats.pickle")
      vocab_filename = os.path.join(feats_id_prefix, "vocab.pickle")
      try:
        print("Loading precomputed features from disk ...")
        gram_feats = pickle.load(open(feats_filename, "rb"))
        gram_feat_names = pickle.load(open(vocab_filename, "rb")) 
      except:
        raise Exception("Something went wrong when loading the n-gram features files ...")

      text_feats_names = gram_feat_names
      text_feats_matrix = gram_feats
      grams_finished = True

    if feats_type in features_dict:
      if feats_type == 'comments_most_reddit':
        df = pd.read_csv(os.path.join(args.data_path, "mbti_enne_pred_comments_most.csv"))
      else:
        df = pd.read_csv(PATH_PREDICTION_FEATS)
      joined = data[["author"]].merge(df, on = "author", how = "left")
      add_feat_names = features_dict[feats_type]
      new_feats = csr_matrix(np.array(joined[add_feat_names]))
      all_feats_names += add_feat_names
      all_feats_list.append(new_feats)
   
  all_feats_matrix = csr_matrix(sparse_hstack(all_feats_list)) if len(all_feats_names) > 0 else None
  if all_feats_matrix is not None:
    assert all_feats_matrix.shape[1] == len(all_feats_names)
    print(all_feats_matrix.shape)
  if text_feats_matrix is not None:
    assert text_feats_matrix.shape[1] == len(text_feats_names)
    print(text_feats_matrix.shape)
  return(text_feats_matrix, text_feats_names, all_feats_matrix, all_feats_names)
  


def spawn_model(hyperparam_combo, reg, label_type, cw, args ):
     if label_type == "classification":
       #model = LinearSVC(C = C, penalty = reg, max_iter = 5000, class_weight = cw) 
       if args.model == "lr":
         model = LogisticRegression(C = hyperparam_combo[0], penalty = reg, class_weight = cw, max_iter = 20000) 
       elif args.model == "et":
         model = ExtraTreesClassifier(n_estimators = hyperparam_combo[0], max_features = hyperparam_combo[1], bootstrap = hyperparam_combo[2], oob_score = hyperparam_combo[3], n_jobs = -1, class_weight = cw)
       elif args.model == "dummy":
         #model = DummyClassifier(strategy = "most_frequent")
         model = DummyClassifier(strategy = "stratified")
       else:
         raise Exception("Unknown model type: " + str(args.model))
       #model = DummyClassifier(strategy="most_frequent")
     elif label_type == "regression":
       if args.model == "lr":
         if reg == "l1":
           model = Lasso(alpha = hyperparam_combo[0], max_iter = 20000)
         elif reg == "l2":
           model = Ridge(alpha = hyperparam_combo[0], max_iter = 20000, solver = 'sag')
         else:
           raise Exception("Unkown regularisation type -- " + str(reg))
       elif args.model == "lr-b":
         model = XGBRegressor(n_estimators=10000, learning_rate=0.002, subsample=0.7, colsample_bytree=0.1, max_depth=5, booster='gbtree', reg_lambda=5.25, reg_alpha=34.85, n_jobs=16, random_state = 42, tree_method = 'gpu_hist')
       elif args.model == "lr-e":
         model = ElasticNet(alpha = hyperparam_combo[0], max_iter = 20000)
       elif args.model == "lr-h":
         model = HuberRegressor(alpha = hyperparam_combo[0], max_iter = 20000)
       elif args.model == "lr-l":
         model = Lasso(alpha = hyperparam_combo[0], max_iter = 20000)
       elif args.model == "lr-s":
         model = SVR(C=hyperparam_combo[0], epsilon=0.2, max_iter=20000, kernel='rbf')
       elif args.model == "lr-class":
         model = LogisticRegression(C = hyperparam_combo[0], penalty = reg, class_weight = cw, max_iter = 20000) 
       elif args.model == "et":
         model = ExtraTreesRegressor(n_estimators = hyperparam_combo[0], max_features = hyperparam_combo[1], bootstrap = hyperparam_combo[2], oob_score = hyperparam_combo[3], n_jobs = -1)
       elif args.model == "dummy":
         model =  DummyRegressor(strategy = "mean")
       else:
         raise Exception("Unknown model type: " + str(args.model))
     else:
       raise Exception("Unkown label type -- " + str(label_type))
     return(model)
          

 

 
# texts -- list of texts to be used (each element of the list is a string -> a one big chunk of comments per user)
# labels -- labels corresponding to above texts
# folds -- a list containing a fold index for each element in the above lists (e.g., for a 5-fold cv, this should be a number 0-4)
def run_baseline(data, data_feats, label_type, run_id_prefix, feat_names, cur_repeat, args, extra_feats, extra_feat_names):
    cw = "balanced" # "balanced" or None
    #reg_types = ["l1","l2"]
    reg_types = ["l2"] # preliminary experiments show that l2 + relatively low number of features in feat sel perform the best, but this might not always be the case
     
    n_base_feats = data_feats.shape[1] if data_feats is not None else 0
    n_extra_feats = extra_feats.shape[1] if extra_feats is not None else 0
    total_n_feats = n_base_feats + n_extra_feats
 
    max_features = 20000
    feat_sel_Ncandidates = [int(percentage * total_n_feats) for percentage in [0.05, 0.1, 0.2, 0.3, 0.4,0.5]]
    if total_n_feats < max_features:
      feat_sel_Ncandidates += ["all"] # if there are not a lot of feats also try a variant with all feats
    else: # on the other hand, dont try more than 20k feats (more than that didn't appear to yield significant benefits in prelim. experiments)
      feat_sel_Ncandidates = [ x for x in feat_sel_Ncandidates if x <= max_features] + [max_features]
    if "lr" in args.model:
      hyperparams = [tuple([2**i]) for i in range(-10,5)] # regularisation strength for the regression models
    elif args.model == "et":
      n_estimators = [100,200,300,400,500]
      mf = ["auto",800] if total_n_feats >= 800 else ["auto"]
      bootstrap = [True]
      oob = [True, False]
      hyperparams = list(itertools.product(*[n_estimators, mf, bootstrap, oob]))
      feat_sel_Ncandidates = ["all"] if total_n_feats < max_features else [max_features]
    elif args.model == "dummy":
      hyperparams = [0]
    else:
      raise Exception("Unknown model type: " + str(args.model))

    print("Starting model run " + run_id_prefix)
     
    valid_indexes = data['label'].notnull()

    # filter out rows with nan vals for extra feats
    if extra_feats is not None:
      ll = np.isnan(extra_feats.todense()).any(axis=1)   
      for ind in range(len(ll)):      
        if ll[ind]: # there is a nan in that row
          valid_indexes[ind] = False
      print("Threw out " + str(np.sum(ll)))
    
    valid_indexes_numbers = np.where(valid_indexes)[0]
     
    filtered_data = data[valid_indexes]
    if data_feats is not None:
      filtered_data_feats = data_feats[valid_indexes_numbers,:]
    if extra_feats is not None:
      filtered_extra_feats = extra_feats[valid_indexes_numbers,:]


    print("[***WARNING***] Removed %d of the authors because the labels were missing." % (len(data) - len(filtered_data)))

    start_time = time.time()

    
    output_topf_path = run_id_prefix + "/topfeats-"+str(cur_repeat)+".csv"

    if not os.path.isdir(run_id_prefix):
      os.mkdir(run_id_prefix)


    if os.path.isfile(output_topf_path):
      os.remove(output_topf_path)

    total_unames, total_true, total_preds, total_folds, total_confidences = [], [], [], [], []
    filter_fs_folds, filter_fs_words = [], []

    folds_to_run = [0,1,2,3,4] if args.specificfold == -1 else [args.specificfold]

    for fold in folds_to_run:
      print("Starting fold " + str(fold))

      test_fold = fold
      val_fold = (fold + 1) % 5

      train_indexes, val_indexes, test_indexes = (filtered_data['fold'] != test_fold) & (filtered_data['fold'] != val_fold), filtered_data['fold'] == val_fold, filtered_data['fold'] == test_fold
      train_indexes_numbers, val_indexes_numbers, test_indexes_numbers = np.where(train_indexes)[0], np.where(val_indexes)[0], np.where(test_indexes)[0]
      
      if data_feats is not None:
        train_feats = filtered_data_feats[train_indexes_numbers]
        val_feats =  filtered_data_feats[val_indexes_numbers]
        test_feats = filtered_data_feats[test_indexes_numbers]
      
      if extra_feats is not None:
        train_extra_feats = filtered_extra_feats[train_indexes_numbers]
        val_extra_feats = filtered_extra_feats[val_indexes_numbers]
        test_extra_feats = filtered_extra_feats[test_indexes_numbers]
      
      
      train_labels = filtered_data[train_indexes]["label"]
      val_labels = filtered_data[val_indexes]["label"]
      test_labels =filtered_data[test_indexes]["label"]
      
      # apply tfidf weighting
      if data_feats is not None:
        print("Applying tfidf for this fold.")
        tfidf = TfidfTransformer(sublinear_tf = True)
        train_feats = tfidf.fit_transform(train_feats)
        val_feats = tfidf.transform(val_feats)
        test_feats = tfidf.transform(test_feats)

      train_unames, test_unames = list(filtered_data[train_indexes]["author"]), list(filtered_data[test_indexes]["author"]) 

      intersection = [v for v in train_unames if v in test_unames]
      if len(intersection) > 0:
        print("RED ALERT, SOME EXAMPLES FROM TRAIN ARE IN TEST, SEND FOR THE SPANISH INQUISITION FORTHWITH!!1one")
        exit(-1)

      val_unames = list(filtered_data[val_indexes]["author"])
      intersection_with_val = [v for v in train_unames if v in val_unames]
      assert len(intersection_with_val) == 0

      best_reg_type, best_feats_N, best_hp, best_score, best_test_preds, best_test_confidences = None, None, None, -1000, None, None       

      
      # some fixes on the extra feats part
      if extra_feats is not None:
        #scaler = StandardScaler(with_mean = False)
        scaler = MinMaxScaler()

        train_extra_feats = csr_matrix(scaler.fit_transform(train_extra_feats.todense()))
        val_extra_feats = csr_matrix(scaler.transform(val_extra_feats.todense()))
        test_extra_feats = csr_matrix(scaler.transform(test_extra_feats.todense()))
       
      
      # combine word feats with all the other feats    
      if data_feats is not None and extra_feats is None:
        combined_train_feats = train_feats
        combined_val_feats = val_feats
        combined_test_feats = test_feats
        combined_feat_names = list(feat_names)
      elif data_feats is None and extra_feats is not None:
        combined_train_feats = csr_matrix(train_extra_feats)
        combined_val_feats = csr_matrix(val_extra_feats)
        combined_test_feats = csr_matrix(test_extra_feats)
        combined_feat_names = list(extra_feat_names)
      elif data_feats is not None and extra_feats is not None:
        for i in range(train_extra_feats.shape[0]):
         if np.isnan(train_extra_feats.todense()[i,:]).any():
           print("NAN FOUND FOR USER :" + train_unames[i])
 
        combined_train_feats = csr_matrix(sparse_hstack([train_feats, csr_matrix(train_extra_feats)]))
        combined_val_feats = csr_matrix(sparse_hstack([val_feats, csr_matrix(val_extra_feats)]))
        combined_test_feats = csr_matrix(sparse_hstack([test_feats, csr_matrix(test_extra_feats)]))
        combined_feat_names = list(feat_names) + list(extra_feat_names)
      else:
        raise Exception("You must supply at least one type of features to use!")
     
      # run the many loops for testing various versions of this and that       
      for feats_N in feat_sel_Ncandidates:
        fs = SelectKBest(chi2, k = feats_N) if label_type == "classification" else SelectKBest(f_regression, k = feats_N) 
        if(feats_N ==0):
          continue
        train_feats_FS = csr_matrix(fs.fit_transform(combined_train_feats, train_labels))
        val_feats_FS = csr_matrix(fs.transform(combined_val_feats))
        test_feats_FS = csr_matrix(fs.transform(combined_test_feats))

        def eval_hp(hype, r,l,c,ar, trf, trl,vlf,vll):
             model = spawn_model(hype, r, l, c, ar)
             model.fit(trf, trl)
             val_preds = model.predict(vlf)
             current_score = f1_score(y_true = vll, y_pred = val_preds, average = "macro") if label_type == "classification" else reg_eval_func(val_labels, val_preds)
             print("Finished for " + str(hype))
             return(model, current_score, hype)
 
        for reg in reg_types:
           train_feats_FS.sort_indices()
           val_feats_FS.sort_indices()

           xval_res = Parallel(n_jobs=12)(delayed(eval_hp)(h, reg, label_type, cw, args, train_feats_FS, train_labels, val_feats_FS, val_labels) for h in hyperparams)

           for model, current_score, hyperparam_combo in xval_res:
             print("Score is %.3f for hyperparams=%s reg=%s and feat_N=%s" % (current_score, str(hyperparam_combo), reg, feats_N))
             if current_score > best_score:
               print("The previous number was the new best xval score!")
               best_score, best_reg_type, best_feats_N, best_hp = current_score, reg, feats_N, hyperparam_combo
               best_test_preds = model.predict(test_feats_FS)
               if label_type == "classification":
                 best_test_confidences = model.predict_proba(test_feats_FS)
                 best_test_confidences = [list(zip(model.classes_, x)) for x in best_test_confidences]
               else:
                 best_test_confidences = best_test_preds
             
      print("FOLD %d best hyperparams were reg=%s, chi2topN=%s, and hyperparams=%s, with score %.3f" % (fold, best_reg_type, best_feats_N, str(best_hp), best_score))
      
      preds = best_test_preds
      confidences = best_test_confidences
      total_unames += test_unames
      total_true += list(test_labels)
      total_preds += list(preds) 
      total_folds += [fold for i in range(len(preds))]
      total_confidences += list(confidences)

      if label_type == "classification":
        print(f1_score(test_labels, preds, average = "macro"))
        print(precision_score(test_labels,preds, average = "macro"))
        print(recall_score(test_labels,preds, average = "macro"))
      else:
        print(reg_eval_func(test_labels, preds))

    PATH_OUTPUT = run_id_prefix + "/preds.csv"
    out_df = pd.DataFrame(list(zip(total_unames, total_folds, total_true, total_preds, total_confidences)), columns =['author', 'fold','true','pred','confidence'])
    out_df.to_csv(PATH_OUTPUT)
    return(PATH_OUTPUT)

# evaluates the model outputs in filename, using F1 score for label_type = "classification" or Pearson correlation for "regression"
# prints mean and stdev of scores across folds, also returns a list of scores in every fold for convenience
def eval_model(filename, label_type, skip_printing = False):
  print("Reading in file :" + filename)
  df = pd.read_csv(filename)
  scores = []
  for fold in range(5):
    if label_type == "classification":
      scores.append(f1_score(df[df["fold"]==fold]["true"], df[df["fold"]==fold]["pred"], average="macro"))
    else:
      pr = list(df[df["fold"]==fold]["pred"])
      tr = list(df[df["fold"]==fold]["true"])
      
      print(" **** FOLD %d ********* min max avg std med are %.3f %.3f %.3f %.3f %.3f" % (fold, np.min(pr), np.max(pr), np.mean(pr), np.std(pr), np.median(pr)))
      md = np.max([abs(pr[i] - tr[i]) for i in range(len(pr))])
      print("Maxdiff = " + str(md))
      print("Manual calculation ...")
      xvals, yvals = list(tr), list(pr)
      mx, my = np.mean(xvals), np.mean(yvals)
      sx, sy, cov = np.std(xvals), np.std(yvals), np.mean([(x - mx) * (y - my) for x,y in zip(xvals,yvals)])
      print("STD pr = " + str(sx))
      print("STD tr = " + str(sy))
      print("cov = " + str(cov)) 
      print("Final pearson score = " + str(cov/(sx*sy)))
      scores.append(reg_eval_func(df[df["fold"]==fold]["true"], df[df["fold"]==fold]["pred"])) # TODO zamijeniti sa scipy.stats.parsonr iako mislim da isto ispada
      print("RMSE is " + str(np.sqrt(np.mean([(x - y)**2 for x,y in zip(xvals, yvals)]))))
  if not skip_printing:
    print("Eval results: ")
    print(np.mean(scores))
    print(np.std(scores))
    print(scores)
  return(scores)

def parse_args(args):
  parser = argparse.ArgumentParser(description='')

  #parser.add_argument("-size", "--size", help="size of the dataset to work on (100,500,700,1000,1500,2000,2500)", type=int)
  parser.add_argument("-data_path", "--data_path", help="Path to the \"data\" folder of this distribution (contains the unfiltered comments, user metadata - author_profiles, fold splits, and some precomputed enneagram/mbti features for the big5 regressions.")
  parser.add_argument("-repeat", "--repeat", help="repeat to work on (0 - 4) or -1 for all repeats (default 0). Each repeat represents a different random split for the five folds. To get results from the paper use repeat 0.",  type=int, default=0)
  parser.add_argument("-label", "--label", help="which column to predict, (any column but most often -- introverted, intuitive, thinking, perceiving, agreeableness, openness, conscientiousness, extraversion, neuroticism, enneagram_type, age, is_female. Additional acceptable values are 'allmbti', 'allbig5', these run the models for multiple columns all at once.", type=str)
  parser.add_argument("-tasktype", "--tasktype", help = "can be 'classification' or 'regression' (default classification)", type=str)
  parser.add_argument("-folds", "--folds", help="which set of folds to use, you probably want 'mbti', 'big5_scores', 'big5_percentiles', 'enneagram', 'age' or 'gender', but any set of folds will work as long as it is compatible with your chosen labels (e.g., don't use 'mbti' folds for predicting a big5 style column, if you do that, something can and WILL go terribly horribly wrong), this parameter also does some prefiltering (see --filters for the nifty details)", type=str)
  parser.add_argument("-feats","--feats", help = "comma separated list of feature types used, can be 1gram, 12gram, 123gram, charngram, style, liwc, age, gender, mbtipred, big5pred (use only one 'gram' style entry if you use more only the first one is considered) ", type=str)
  parser.add_argument("-variant","--variant", help="string describing a variant of the experiment, such as FemaleOnly or Over21RegressionPercentilesOnly", type=str)
  parser.add_argument("-model", "--model", help="optional, which model to use, 'lr' (logistic/linear regression), or 'et' (extratrees classifier/regressor) or 'dummy' (most frequent class for classification or train set mean for regression), default is lr", default = "lr", type=str)
  parser.add_argument("-specificfold", "--specificfold", help = "optional, speicifies which particular fold to run (single number 0 to 4), or -1 to run all folds, default is -1", default=-1, type=int)
  parser.add_argument("-topfeats", "--topfeats", help = "optional, number of top features from feature selection to dump to file, -1 to dump all used feats, default is -1", default=-1, type=int)
  return parser.parse_args(args)

def main(args):
   args = parse_args(args)

   repeats_to_run = [args.repeat] if args.repeat in [0,1,2,3,4] else [0,1,2,3,4]   
   feats_type = args.feats # unigram, charngram or bitrigram
   task_variant = args.variant # any sufix with additional info describing optional variants, for example "FemaleOnly" or "PercentileScalesOnly"
   

   #outfile_prefix = "./res/" + str(args.variant)

   if args.label == "allmbti":
     labels = ["introverted", "intuitive","thinking","perceiving"]
   elif args.label == "allbig5":
     labels = ["agreeableness","openness","conscientiousness","extraversion","neuroticism"]     
   else:
     labels = [args.label]

   unm = pickle.load(open(os.path.join(args.data_path, "unames.pickle"), "rb"))
   txt = []
   

   for label_name in labels:
     task_prefix = os.path.join("./res/", "-".join([label_name, task_variant]))

     label_type = args.tasktype
     FOLD_GRP = args.folds
     print("Starting experiment for *** " + label_name  + " *** ...") # TODO add printout of arguments
     experiment_start_time = time.time()   
     for current_repeat in repeats_to_run:
       data = load_data(unm, txt, os.path.join(args.data_path, "author_profiles.csv"), label_name, label_type, FOLD_GRP, current_repeat, args)     
       data_feats, feat_names, extra_feats, extra_feat_names = precompute_or_load_feats(data, "./data", args)
       preds_filename = run_baseline(data, data_feats, label_type, task_prefix, feat_names, current_repeat, args, extra_feats, extra_feat_names)
       eval_model(preds_filename, label_type)
     print("Finished experiment, time required was " + str(time.time() - experiment_start_time))


if __name__ == "__main__":
    main(sys.argv[1:])
