import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from pipeline_processor.record import *


def get_config(data_name, model_name, prompt_type):

    if "MSVD" == data_name:
        path_qa = "./data/sample_qa/msvd_only_test_5percent.csv"
        path_video = "../../../Users/oldnonstop@gmail.com/vgrid/YouTubeClips/%s.avi"
    elif "MSRVTT" == data_name:
        path_qa = "./data/sample_qa/msrvtt_only_val_2.5percent.csv"
        path_video = "/dbfs/wonkyun/TrainValVideo/video%s.mp4"
    elif "NEXTQA" == data_name:
        path_qa = "./data/sample_qa/nextqa_only_test_5percent.csv"
        path_video = "/dbfs/wonkyun/nextqa/videos/%s.mp4"
    elif "ACTIVITY" == data_name:
        path_qa = "./data/sample_qa/activitynet_only_test_5percent_ver_videoexist.csv"
        path_video = "/dbfs/wonkyun/activitynet_dropbox/videos/test/all/%s.mp4"

    user_prompt = ""
    func_user_prompt = lambda x: round(math.sqrt(x))

    if PromptType.GRIDVIEW_ORDER == prompt_type:
        if "NEXTQA" == data_name:
            user_prompt = "USER: <image>\n The provided image arranges key frames from a video in a grid view. They are arranged in chronological order, holding temporal information from the top left to the bottom right. You need to choose one of the following five options to answer the question, '%s?' : '%s', '%s', '%s', %s', '%s'. Please give me a brief answer.\nASSISTANT:"
            func_user_prompt = lambda prompt, row: prompt % (
                row["question"],
                row["a0"],
                row["a1"],
                row["a2"],
                row["a3"],
                row["a4"],
            )
        else:
            user_prompt = "USER: <image>\nThe provided image arranges key frames from a video in a grid view. They are arranged in chronological order, holding temporal information from the top left to the bottom right. You need to answer the questions '%s?'\nASSISTANT:"
            func_user_prompt = lambda prompt, row: prompt % (row["question"])
    elif PromptType.ONLY_QUESTION == prompt_type:
        if "NEXTQA" == data_name:
            user_prompt = "USER: <image>\n You need to choose one of the following five options to answer the question, '%s?' : '%s', '%s', '%s', %s', '%s'. Please give me a brief answer.\nASSISTANT:"
            func_user_prompt = lambda prompt, row: prompt % (
                row["question"],
                row["a0"],
                row["a1"],
                row["a2"],
                row["a3"],
                row["a4"],
            )
        else:
            user_prompt = (
                "USER: <image>\nYou need to answer the questions '%s?'\nASSISTANT:"
            )
            func_user_prompt = lambda prompt, row: prompt % (row["question"])

    dir = "/dbfs/wonkyun/llava_pipeline_result1/%s/" % (data_name)
    experiment_name = "/Users/oldnonstop@gmail.com/videogrid_%s_%s" % (
        data_name,
        model_name,
    )

    return (
        path_qa,
        path_video,
        dir,
        user_prompt,
        func_user_prompt,
        experiment_name,
    )


if __name__ == "__main__":

    data_name = "MSVD"
    model_name = "Llava"
    prompt_type = PromptType.GRIDVIEW_ORDER

    path_qa, path_video, dir, user_prompt, func_user_prompt, experiment_name = (
        get_config(data_name, model_name, prompt_type)
    )

    print(path_qa)
    print(path_video)
    print(dir)
    print(user_prompt)
    print(func_user_prompt)
    print(experiment_name)

    data_name = "NEXTQA"
    model_name = "Llava"
    prompt_type = PromptType.ONLY_QUESTION

    path_qa, path_video, dir, user_prompt, func_user_prompt, experiment_name = (
        get_config(data_name, model_name, prompt_type)
    )

    print(path_qa)
    print(path_video)
    print(dir)
    print(user_prompt)
    print(func_user_prompt)
    print(experiment_name)
