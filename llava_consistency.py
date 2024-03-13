import sys, os
from io import BytesIO

from model_processor.llava_model_processor import *
from vision_processor.fps_gridview_processor import *
from pipeline_processor.llava_pipeline import *
from evaluation.gpt3_consistency_utils import *
import pandas as pd


def main():
    path_qa1 = "./data/quantitative_eval/consistency_qa1.csv"
    path_qa2 = "./data/quantitative_eval/consistency_qa2.csv"
    path_video = "/data2/wslee/VideoGrid/data/Test_Videos/%s.avi"
    dir1 = "/data2/wslee/VideoGridPaper/results/Consistency1/"
    dir2 = "/data2/wslee/VideoGridPaper/results/Consistency2/"

    user_prompt = "USER: <image>\nThe provided image arranges key frames from a video in a grid view. They are arranged in chronological order, holding temporal information from the top left to the bottom right. You need to answer the questions '%s?'. \nASSISTANT:"

    interval = 2
    quality = 95
    func_user_prompt = lambda prompt, row: prompt % (row["question"])
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
    frame_fixed_number = 4
    frame_selection_on_range = FrameSelectionOnRange.HIGH_CONTRAST

    """
    Next-QA
    
    user_prompt = "USER: <image>\nThe provided image arranges key frames from a video in a grid view. They are arranged in chronological order, holding temporal information from the top left to the bottom right. You need to choose one of the following five options to answer the question, '%s?' : '%s', '%s', '%s', %s', '%s'. Please give me a brief answer. \nASSISTANT:"
    func_user_prompt = lambda prompt, row: prompt % (row["question"], row["a0"], row["a1"], row["a2"], row["a3"], row["a4"])
    
    """

    llavaPipeline = LlavaPipeline(path_qa1, path_video, dir=dir1)
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
    df_merged1, path_df_merged1 = llavaPipeline.do_pipeline()

    llavaPipeline.reset_pipeline(path_qa2, path_video, dir2)
    df_merged2, path_df_merged2 = llavaPipeline.do_pipeline()

    api_key = ""
    num_core_gpt3 = 40
    gpt3_dir = (
        "/data2/wslee/VideoGridPaper/results_gpt3_evaluation/5_Consistency/"
        + path_df_merged1.split("/")[-2]
        + "/"
    )

    df_qa, path_merged = gpt3_consistency_parallel_processing(
        df_merged1, df_merged2, gpt3_dir, num_core_gpt3, api_key
    )

    print("final file path : " + path_merged)
    print(df_qa.head())


if __name__ == "__main__":
    main()
