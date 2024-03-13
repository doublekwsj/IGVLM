Video Grid Paper Repository



Models aligning large language and vision modalities offer significant advantages in addressing multimodal tasks. However, tackling video tasks remains challenging. To address video-related challenges, it is necessary not only to leverage diverse video data but also to extract video embeddings and align them with language models through the training. This paper introduces VideoGrid, a novel approach leveraging existing LLM with vision encoder (VLLM) to solve Video Tasks without additional model training. The proposed method involves a pre-processing step that represent a image by adding temporal information from a video. The VLLM, grounded in the reasoning abilities of language models, is then proposed to understand videos based on the added temporal and spatial information in the images. VideoGrid exhibits comparable video question-answering performance to existing Video Language Models, achieved without specific model training.
