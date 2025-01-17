# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
import paddlemix.models.jina_clip_v2 import *
from PIL import Image


# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']

image_path = 'paddlemix/demo_images/jina_clip_v2_bench.jpg'

# Choose a matryoshka dimension, set to None to get the full 1024-dim vectors
truncate_dim = 512

# Encode text and images
# 图像编码器
text_embeddings = model.encode_text(sentences, truncate_dim=truncate_dim)

# 文本编码器
image_embeddings = model.encode_image(
    image_urls, truncate_dim=truncate_dim
)  # also accepts PIL.Image.Image, local filenames, dataURI

# Encode query text
query = 'beautiful sunset over the beach' # English
query_embeddings = model.encode_text(
    query, task='retrieval.query', truncate_dim=truncate_dim
)

# Text to Image
print('En -> Img: ' + str(query_embeddings @ image_embeddings[0].T))
# Image to Image
print('Img -> Img: ' + str(image_embeddings[0] @ image_embeddings[1].T))
# Text to Text
print('En -> Ar: ' + str(query_embeddings @ text_embeddings[0].T))
print('En -> Zh: ' + str(query_embeddings @ text_embeddings[1].T))
print('En -> Fr: ' + str(query_embeddings @ text_embeddings[2].T))
print('En -> De: ' + str(query_embeddings @ text_embeddings[3].T))
print('En -> Gr: ' + str(query_embeddings @ text_embeddings[4].T))
print('En -> Hi: ' + str(query_embeddings @ text_embeddings[5].T))
print('En -> It: ' + str(query_embeddings @ text_embeddings[6].T))
print('En -> Jp: ' + str(query_embeddings @ text_embeddings[7].T))
print('En -> Ko: ' + str(query_embeddings @ text_embeddings[8].T))



