import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
import argparse

def spearman_corr(y_t, y_p):
    """
    Calculate the Spearman correlation coefficient between two arrays.

    Args:
        y_t (array-like): True values.
        y_p (array-like): Predicted values.

    Returns:
        float: Spearman correlation coefficient.

    """
    return spearmanr(y_p, y_t)[0]


def my_pearson(y_t, y_p):
    """
    Calculate the Pearson correlation coefficient between two arrays.

    Args:
        y_t (array-like): True values.
        y_p (array-like): Predicted values.

    Returns:
        float: Pearson correlation coefficient.

    """
    return pearsonr(y_t, y_p)[0]


def my_f1(y_t, y_p):
    """
    Calculate the F1 score between two arrays.

    Args:
        y_t (array-like): True values.
        y_p (array-like): Predicted values.

    Returns:
        float: F1 score.

    """
    return f1_score(y_true=y_t, y_pred=y_p, average="macro")


def my_rmse(y_t, y_p):
    """
    Calculate the root mean squared error (RMSE) between two arrays.

    Args:
        y_t (array-like): True values.
        y_p (array-like): Predicted values.

    Returns:
        float: Root mean squared error (RMSE).

    """
    return np.sqrt(mean_squared_error(y_t, y_p))


def calculate_scores(variant):
    """
    Calculate scores for different tasks and evaluation metrics.

    Args:
        variant (str): The variant of the scores to calculate. Should be either "LR-N" or "LR-NP".

    Raises:
        Exception: If the variant is not "LR-N" or "LR-NP".

    """
    if variant == "LR-N":
        table_rows = [("introverted", my_f1, "MBTI, I vs E (F1 score)"),
                      ("intuitive", my_f1, "MBTI, N vs P (F1 score)"),
                      ("thinking", my_f1, "MBTI, T vs F (F1 score)"),
                      ("perceiving", my_f1, "MBTI, P vs J (F1 score)"),
                      ("enneagram_type", my_f1, "Enneagram (F1 score)"),
                      ("is_female", my_f1, "Gender (F1 score)"),
                      ("region", my_f1, "Region (F1 score)"),
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

    with open("results_output.txt", "a") as file_object:
        for task_name, eval_metric, pretty_task_name in table_rows:
            df = pd.read_csv(os.path.join("./res/", "-".join([task_name, variant]), "preds.csv"))
            scores = [eval_metric(y_t=df[df["fold"] == f]["true"], y_p=df[df["fold"] == f]["pred"]) for f in range(5)]
            rmse = [my_rmse(y_t=df[df["fold"] == f]["true"], y_p=df[df["fold"] == f]["pred"]) for f in range(5)]
            output = f"{pretty_task_name} -- {np.mean(scores):.3f} +- {np.std(scores):.3f} --- {np.mean(rmse):.3f}"
            print(output)
            file_object.write(output)
            file_object.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["LR-N", "LR-NP"], help="Variant for calculation")
    args = parser.parse_args()

    calculate_scores(args.variant)


if __name__ == "__main__":
    main()
