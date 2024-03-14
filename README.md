

<h2 align="center"> <a>IG-VLM: An Image Grid Can Be Worth a Video: Zero-shot Video Question Answering Using a VLM</a></h2>
Stimulated by the sophisticated reasoning capabilities of recent Large Language Models(LLMs), a variety of strategies for bridging video modality have been devised. A prominent strategy involves Video Language Models(VideoLMs), which train a learnable interface with video data to connect advanced vision encoders with LLMs. Recently, an alternative strategy has surfaced, employing readily available foundation models, such as VideoLMs and LLMs, across multiple stages for modality bridging. In this study, we introduce a simple yet novel strategy where only a single Vision Language Model (VLM) is utilized. Our starting point is the plain insight that a video comprises a series of images, or frames, interwoven with temporal information. The essence of video comprehension lies in adeptly managing the temporal aspects along with the spatial details of each frame. Initially, we transform a video into a single composite image by arranging multiple frames in a grid layout. The resulting single image is termed as an image grid. This format, while maintaining the appearance of a solitary image, effectively retains temporal information within the grid structure. Therefore, the image grid approach enables direct application of a single high-performance VLM without necessitating any video-data training. Our extensive experimental analysis across ten zero-shot video question answering benchmarks, including five open-ended and five multiple-choice benchmarks, reveals that the proposed Image Grid Vision Language Model (IG-VLM) surpasses the existing methods in nine out of ten benchmarks.

## Requirements and Installation
* Pytorch == 2.2.0
* transformers==4.36.2
* Install required packages : pip install -r requirements.txt
* Please make sure that pytorch version to reproduce our results. 


## Inference and Evaluation
 논문에서 실험한 LLaVA v1.6 7b/13b/34b, GPT-4V에 IG-VLM을 적용해서, 결과를 재현할 수 있는 코드를 제공한다. 각 모델별로 open-ended VQA(MSVD-QA, MSRVTT-QA, ActivityNet-QA, TGIF-QA), Text Generation Performance VQA(CI, DO, CU, TU, CO), multiple-choice VQA (NExT-QA, STAR, TVQA, EgoSchema, IntentQA)를 실험할 수 있는 파일이 각 기 제공된다. 
 * 각 Benchmark 실험을 위해 Data Download와 QA pair sheet를 준비한다. 
 * 이때, QA pair sheet는 아래와 같은 format을 가지며, csv파일로 변환해야 한다. 
 * LLaVA v1.6 with IG-VLM을 실험하는 경우, 아래와 같은 명령어로 진행할 수 있다. llm_size parameter를 이용해 7b, 13b, 34b 모델 중 하나를 선택할 수 있다.
 ```bash
 # Open-ended video question answering (MSVD-QA, MSRVTT-QA, ActivityNet-QA and TGIF-QA)
 python eval_llava_openended.py --path_qa_pair_csv ./data/openended_qa/activitynet.csv --path_video /data/activitynet/videos/%s.mp4 --path_result ./result_activitynet/ --api_key {api_key}
 ```
 ```bash
 # Text generation performance (CI, DO, CU, TU and CO)
 python eval_llava_multiplechoice.py --path_qa_pair_csv ./data/multiple_choice/tvqa.csv --path_video /data/TVQA/videos/%s.mp4 --path_result ./result_tvqa/
 ```
 ```bash
 # Multiple-choice VQA (NExT-QA, STAR, TVQA, IntentQA and EgoSchema)
 python eval_llava_textgeneration_openended.py --path_qa_pair_csv ./data/text_generation_benchmark/generic_qa.csv --path_video /data/activitynet/videos/%s.mp4 --path_result ./result_textgeneration/ --api_key {api_key}
 ```
 * GPT-4V with IG-VLM을 실험하는 경우, 아래와 같은 명령어로 진행할 수 있다. gpt4 vision api를 사용하기 때문에, 비용이 많이 발생할 수 있다. 
 ```bash
 # Open-ended video question answering (MSVD-QA, MSRVTT-QA, ActivityNet-QA and TGIF-QA)
 python eval_gpt4v_openended.py --path_qa_pair_csv ./data/openended_qa/msvd_qa.csv --path_video /data/msvd/videos/%s.avi --path_result ./result_activitynet_gpt4/ --api_key {api_key}
 ```
 ```bash
 # Text generation performance (CI, DO, CU, TU and CO)
 python eval_gpt4v_textgeneration_openended.py --path_qa_pair_csv ./data/text_generation_benchmark/generic_qa.csv --path_video /data/activitynet/videos/%s.mp4 --path_result ./result_textgeneration_gpt4/ --api_key {api_key}
 ```
 ```bash
 # Multiple-choice VQA (NExT-QA, STAR, TVQA, IntentQA and EgoSchema)
 python eval_gpt4v_multiplechoice.py --path_qa_pair_csv ./data/multiple_choice_qa/EgoSchema.csv --path_video /data/EgoSchema/videos/%s.mp4 --path_result ./result_egoschema_gpt4/ --api_key {api_key}
 ```
```

