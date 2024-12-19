export NCCL_DEBUG=""
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install pytorch==2.3.0 torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install torchvision torchaudio
pip install vllm==0.5.0
pip install deepspeed
pip install nltk
pip install numerize
pip install rouge-score
pip install torchtyping
pip install rich
pip install accelerate
pip install datasets
pip install sentencepiece
pip install protobuf
pip install peft
# pip install megatron