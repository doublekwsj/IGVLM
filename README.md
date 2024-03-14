

<h2 align="center"> <a>IG-VLM: An Image Grid Can Be Worth a Video: Zero-shot Video Question Answering Using a VLM</a></h2>
Stimulated by the sophisticated reasoning capabilities of recent Large Language Models(LLMs), a variety of strategies for bridging video modality have been devised. A prominent strategy involves Video Language Models(VideoLMs), which train a learnable interface with video data to connect advanced vision encoders with LLMs. Recently, an alternative strategy has surfaced, employing readily available foundation models, such as VideoLMs and LLMs, across multiple stages for modality bridging. In this study, we introduce a simple yet novel strategy where only a single Vision Language Model (VLM) is utilized. Our starting point is the plain insight that a video comprises a series of images, or frames, interwoven with temporal information. The essence of video comprehension lies in adeptly managing the temporal aspects along with the spatial details of each frame. Initially, we transform a video into a single composite image by arranging multiple frames in a grid layout. The resulting single image is termed as an image grid. This format, while maintaining the appearance of a solitary image, effectively retains temporal information within the grid structure. Therefore, the image grid approach enables direct application of a single high-performance VLM without necessitating any video-data training. Our extensive experimental analysis across ten zero-shot video question answering benchmarks, including five open-ended and five multiple-choice benchmarks, reveals that the proposed Image Grid Vision Language Model (IG-VLM) surpasses the existing methods in nine out of ten benchmarks.

## Requirements and Installation
* Pytorch == 2.2.0
* transformers==4.36.2
* Install required packages : pip install -r requirements.txt
* Please make sure that pytorch version to reproduce our results. 


