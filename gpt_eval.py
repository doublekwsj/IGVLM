import os
import argparse
import pandas as pd
from evaluation.gpt3_evaluation_utils import *
from evaluation.gpt3_consistency_utils import *


def main():
    parser = argparse.ArgumentParser(description="Process GPT-3 evaluation.")
    parser.add_argument("--df_merged_path", type=str, help="Path to the df_qa CSV file")
    parser.add_argument(
        "--output_path", type=str, help="Path to the gpt3 result CSV file"
    )

    args = parser.parse_args()

    df_merged = pd.read_csv(args.df_merged_path)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key not set in environment variables")

    num_core_gpt3 = 40
    gpt3_dir = args.output_path + args.df_merged_path.split("/")[-2] + "/"

    gpt_eval_type = EvaluationType.DEFAULT
    # gpt_eval_type = EvaluationType.CORRECTNESS
    # gpt_eval_type = EvaluationType.DETAILED_ORIENTATION
    # gpt_eval_type = EvaluationType.CONTEXT
    # gpt_eval_type = EvaluationType.TEMPORAL

    df_qa, path_merged = gpt3_parallel_processing(
        df_merged,
        gpt3_dir,
        num_core_gpt3,
        api_key,
        gpt_eval_type,
        evaluation_method=process_gpt3_evaluation_v2,
    )

    # df_merged2 = pd.read_csv(args.df_merged_path.replace("consistency1","consistency2"))
    # df_qa, path_merged = gpt3_consistency_parallel_processing(
    #    df_merged, df_merged2, gpt3_dir, num_core_gpt3, api_key
    # )

    print("final file path : " + path_merged)
    print(df_qa.head())
    yes_count = df_qa[df_qa["gpt3_pred"] == "yes"].shape[0]
    print(yes_count / df_qa.shape[0])
    print(df_qa["gpt3_score"].mean())


if __name__ == "__main__":
    main()
