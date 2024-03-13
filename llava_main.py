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
    dir = "/dbfs/wonkyun/llava_pipeline_result/MSVD/"

    # Open-Ended QA
    # LLaVA v1.6 7B,13B prompt
    user_prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? \nASSISTANT:\nAnswer: In the video,"
    # CogAgent prompt
    # user_prompt = "USER: <img><Image></img>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? \nAnswer: In the video,"
    # LLaVA v1.6 34B prompt
    # user_prompt = "<|im_start|>system\n Answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? <|im_end|>\n<|im_start|>assistant\nAnswer: In the video,"
    func_user_prompt = lambda prompt, row: prompt % (row["question"])
    # user_prompt = "USER: <image>\nThe provided image arranges key frames from a video in a grid view. They are arranged in chronological order, holding temporal information from the top left to the bottom right. You need to choose one of the following five options to answer the question, '%s?' : '%s', '%s', '%s', '%s', '%s'. Please give me a brief answer. \nASSISTANT:"
    # func_user_prompt = lambda prompt, row: prompt % (row["question"], row["a0"], row["a1"], row["a2"], row["a3"], row["a4"])

    # Multiple-Choice QA
    # LLaVA v1.6 7B,13B prompt
    # user_prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Question: %s?\n A:%s. B:%s. C:%s. D:%s. E:%s. \n Select the correct answer from the options(A,B,C,D,E). \nASSISTANT: \nAnswer:"
    # CogAgent
    # user_prompt = "USER: <img><Image></img>\nThe provided image arranges key frames from a video in a grid view. Question: %s? A:%s. B:%s. C:%s. D:%s. E:%s.\n Select the correct answer from the options(A,B,C,D,E). \nASSISTANT: \nAnswer:"
    # LLaVA v1.6 34B prompt
    # user_prompt = "<|im_start|>system\n Select correct option to answer the question.<|im_end|>\n<|im_start|>user\n <image>\n Question: %s? A:%s. B:%s. C:%s. D:%s. E: %s. Select the correct answer from the options. <|im_end|>\n<|im_start|>assistant\nAnswer:"

    interval = 1
    quality = 95
    calculate_max_row = lambda x: round(math.sqrt(x))
    has_condition = True
    condition_list = [8]
    condition_interval = [0.5]
    add_border = False
    border_thickness = 5

    # added parameters for experiment
    prompt_type = (
        PromptType.GRIDVIEW_ORDER
    )  # this is for mlflow logging. you need to change prompt variables.
    frame_extraction_type = FrameExtractionType.UNIFORM
    grid_order_type = GridOrder.HORIZONTAL
    resolution_aspect_ratio = ResolutionAspectRatio.SQRT
    # frame_fixed_number = -1
    frame_fixed_number = 4
    frame_selection_on_range = FrameSelectionOnRange.HIGH_CONTRAST

    # added parameters for gpt3 evaluation
    # 0: default, 1: correctness, 2: detailed orientation, 3: context, 4: temporal
    gpt_eval_type = EvaluationType.DEFAULT

    llavaPipeline = LlavaPipeline(path_qa, path_video, dir=dir)
    llavaPipeline.set_component(
        user_prompt,
        interval=interval,
        quality=quality,
        func_user_prompt=func_user_prompt,
        calculate_max_row=calculate_max_row,
        has_condition=has_condition,
        condition_list=condition_list,
        condition_interval=condition_interval,
        add_border=add_border,
        border_thickness=border_thickness,
        prompt_type=prompt_type,
        frame_extraction_type=frame_extraction_type,
        grid_order_type=grid_order_type,
        resolution_aspect_ratio=resolution_aspect_ratio,
        frame_fixed_number=frame_fixed_number,
        frame_selection_on_range=frame_selection_on_range,
    )
    df_merged, path_df_merged = llavaPipeline.do_pipeline()

    api_key = ""
    num_core_gpt3 = 40
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

    #################################################################################################
    # video file과 실험 셋팅 다르게 해서 돌리기
    #################################################################################################
    new_path_qa = "??"
    new_video_path_format = "??"

    llavaPipeline.reset_pipeline(new_path_qa, new_video_path_format)
    llavaPipeline.set_component(
        user_prompt="Please describe it", interval=interval + 1, quality=quality - 10
    )
    df_merged_other_setting, path_df_merged_other_setting = llavaPipeline.do_pipeline()
    df_qa, path_merged = gpt3_parallel_processing(
        df_merged_other_setting, gpt3_dir, num_core_gpt3, api_key, gpt_eval_type
    )


if __name__ == "__main__":
    main()
