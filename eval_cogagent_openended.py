import sys, os
import time
from io import BytesIO

from vision_processor.fps_gridview_processor import *
from pipeline_processor.llava_pipeline import *
from pipeline_processor.gpt4_pipeline import *
from evaluation.gpt3_evaluation_utils import *
import pandas as pd


def main():
    path_qa = "./data/sample_qa/msvd_only_test_5percent.csv"
    path_video = "../../../Users/oldnonstop@gmail.com/vgrid/YouTubeClips/%s.avi"
    dir = "./tmp/llava_pipeline_result/MSVD/"

    # Open-Ended QA
    # LLaVA v1.6 7B,13B prompt
    user_prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? \nASSISTANT:\nAnswer: In the video,"
    # CogAgent prompt
    # user_prompt = "USER: <img><Image></img>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? \nAnswer: In the video,"
    # LLaVA v1.6 34B prompt
    # user_prompt = "<|im_start|>system\n Answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? <|im_end|>\n<|im_start|>assistant\nAnswer: In the video,"

    func_user_prompt = lambda prompt, row: prompt % (row["question"])

    frame_fixed_number = 6
    gpt_eval_type = EvaluationType.DEFAULT

    llavaPipeline = LlavaPipeline(path_qa, path_video, dir=dir)
    llavaPipeline.set_component(
        user_prompt,
        func_user_prompt=func_user_prompt,
        frame_fixed_number=frame_fixed_number,
    )
    df_merged, path_df_merged = llavaPipeline.do_pipeline()

    api_key = ""
    num_core_gpt3 = 20
    gpt3_dir = (
        "/data2/wslee/VideoGridPaper/results_gpt3_evaluation/1_Correctness/"
        + path_df_merged.split("/")[-2]
        + "/"
    )

    df_qa, path_merged = gpt3_parallel_processing(
        df_merged, gpt3_dir, num_core_gpt3, api_key, gpt_eval_type
    )

    print("final file path : " + path_merged)
    print(df_qa.head())


if __name__ == "__main__":
    main()
