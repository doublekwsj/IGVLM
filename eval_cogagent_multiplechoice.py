import sys, os
import time
from io import BytesIO
import argparse
import re
import uuid

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from vision_processor.fps_gridview_processor import *

from pipeline_processor.cogagent_pipeline import *
from pipeline_processor.gpt4_pipeline import *
from evaluation.direct_answer_eval import *


def infer_and_eval_model(args):
    path_qa = args.path_qa_pair_csv
    path_video = args.path_video
    path_result_dir = args.path_result

    user_prompt = get_prompt()
    frame_fixed_number = 6

    # NExT-QA, TVQA, IntentQA, EgoSchema
    func_user_prompt = lambda prompt, row: prompt % (
        row["question"],
        row["a0"],
        row["a1"],
        row["a2"],
        row["a3"],
        row["a4"],
    )

    # In case of STAR benchamrk, use the following codes and select prompt according to llm size.
    """
    func_user_prompt = lambda prompt, row: prompt % (
        row["question"],
        row["a0"],
        row["a1"],
        row["a2"],
        row["a3"]
    )
    # 7b, 13b
    prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Question: %s?\n A:%s. B:%s. C:%s. D:%s. \n Select the correct answer from the options(A,B,C,D). \nASSISTANT: \nAnswer:"
    # 34b 
    prompt = "<|im_start|>system\n Select correct option to answer the question.<|im_end|>\n<|im_start|>user\n <image>\n Question: %s? A:%s. B:%s. C:%s. D:%s. Select the correct answer from the options. <|im_end|>\n<|im_start|>assistant\nAnswer:"
    """

    print("loading CogAgent")

    cogagentPipeline = CogagentPipeline(
        path_qa,
        path_video,
        dir=path_result_dir,
    )
    cogagentPipeline.set_component(
        user_prompt,
        frame_fixed_number=frame_fixed_number,
        func_user_prompt=func_user_prompt,
    )
    df_merged, path_df_merged = cogagentPipeline.do_pipeline()

    print("cogagent prediction result : " + path_df_merged)
    print("start multiple-choice evaluation")

    eval_multiple_choice(df_merged)


def get_prompt():
    user_prompt = "USER: <img><Image></img>\nThe provided image arranges key frames from a video in a grid view. Question: %s? A:%s. B:%s. C:%s. D:%s. E:%s.\n Select the correct answer from the options(A,B,C,D,E). \nASSISTANT: \nAnswer:"
    return user_prompt


def validate_video_path(filename):
    pattern = r"\.(avi|mp4|mkv|gif|webm)$"  # %s.avi 또는 %s.mp4 형식을 따르는지 확인하는 정규 표현식
    if not re.search(pattern, filename):
        raise argparse.ArgumentTypeError(
            f"No valid video path. You must include %s and the extension of video file. (e.g., /tmp/%s.mp4)"
        )
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA v1.6 with IG-VLM")
    parser.add_argument(
        "--path_qa_pair_csv",
        type=str,
        required=True,
        help="path of question and answer. It should be csv files",
    )
    parser.add_argument(
        "--path_video",
        type=validate_video_path,
        required=True,
        metavar="/tmp/%s.mp4",
        help="path of video files. You must include string format specifier and the extension of video file.",
    )
    parser.add_argument(
        "--path_result", type=str, required=True, help="path of output directory"
    )

    args = parser.parse_args()

    infer_and_eval_model(args)
