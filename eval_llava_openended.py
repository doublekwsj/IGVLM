import sys, os
import time
from io import BytesIO
import argparse
import re

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from vision_processor.fps_gridview_processor import *

from pipeline_processor.llava_pipeline import *
from pipeline_processor.gpt4_pipeline import *
from evaluation.gpt3_evaluation_utils import *


def get_llava_and_prompt(llm_size):
    if llm_size in ["7b", "13b"]:
        prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? \nASSISTANT:\nAnswer: In the video,"
        model_name = "llava-v1.6-vicuna-%s" % (llm_size)
    else:
        prompt = "<|im_start|>system\n Answer the question. <|im_end|>\n<|im_start|>user\n <image>\nThe provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s? <|im_end|>\n<|im_start|>assistant\nAnswer: In the video,"
        model_name = "llava-v1.6-%s" % (llm_size)
    return model_name, prompt


def infer_and_eval_model(args):
    path_qa = args.path_qa_pair_csv
    path_video = args.path_video
    path_result_dir = args.path_result
    llm_size = args.llm_size
    num_core_eval = args.num_core_eval
    api_key = args.api_key

    model_name, user_prompt = get_llava_and_prompt(llm_size)

    print(args)
    print(model_name)
    print(user_prompt)

    func_user_prompt = lambda prompt, row: prompt % (row["question"])
    frame_fixed_number = 6

    print("loading [%s]" % (model_name))

    llavaPipeline = LlavaPipeline(
        model_name,
        path_qa,
        path_video,
        dir=path_result_dir,
    )
    llavaPipeline.set_component(
        user_prompt,
        func_user_prompt=func_user_prompt,
        frame_fixed_number=frame_fixed_number,
    )
    df_merged, path_df_merged = llavaPipeline.do_pipeline()

    print("start gpt3-evaluation")

    api_key = ""
    gpt3_dir = path_result_dir + "results_gpt3_evaluation/"

    df_qa, path_merged = gpt3_parallel_processing(
        df_merged, gpt3_dir, num_core_eval, api_key
    )

    print("final file path : " + path_merged)
    print(df_qa.head())
    yes_count = df_qa[df_qa["gpt3_pred"] == "yes"].shape[0]
    score = df_qa["gpt3_score"].describe()["mean"]

    print("Acc : %s" % (str(yes_count / df_qa.shape[0])))
    print("Score : %s" % (str(score)))


def validate_llm_size(type_llm_size):
    if type_llm_size not in {"7b", "13b", "34b"}:
        raise argparse.ArgumentTypeError(f"No valid LLM size.")
    return type_llm_size


def validate_video_path(filename):
    pattern = r"^%s\.(avi|mp4|mkv|gif)$"  # %s.avi 또는 %s.mp4 형식을 따르는지 확인하는 정규 표현식
    if not re.match(pattern % filename, filename):
        raise argparse.ArgumentTypeError(
            f"No valid video path. You must include %s and the extension of video file."
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
        type=str,
        required=True,
        metavar="/tmp/%s.mp4",
        help="path of video files. You must include string format specifier and the extension of video file.",
    )
    parser.add_argument(
        "--path_result", type=str, required=True, help="path of output directory"
    )

    parser.add_argument(
        "--llm_size",
        type=validate_llm_size,
        default="7b",
        help="You can choose llm size of LLaVA. 7b | 13b | 34b",
    )

    parser.add_argument(
        "--num_core_eval",
        type=int,
        default=5,
        help="the number of core for gpt3 evaluation",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="api key for gpt-3 evaluation",
    )
    args = parser.parse_args()

    infer_and_eval_model(args)
