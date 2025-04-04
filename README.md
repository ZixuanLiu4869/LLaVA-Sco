# 🔥 LLaVA-SCo: Teach Vision Language Models to Self-Correct

LLaVA-SCo introduces a self-correction stage into the reasoning pipeline of Vision-Language Models (VLMs), enabling models to refine their outputs through structured reflection. Our approach builds on the foundation of [LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT) and demonstrates improvements on complex visual reasoning tasks.




## 📁 Dataset
You can download our dataset from:

👉 [Google Drive Download](https://drive.google.com/file/d/1937sO0WJwKIAT1661wk7jCgw4-b_kAyO/view?usp=sharing)

To generate your own dataset, run dataset_generation.py.

Before running, update the following:

1. Line 16: Replace with your OpenAI API key.

2. Line 17: Set the OpenAI API endpoint (e.g., for Azure users).

3. Line 127: Set the path to your image directory.

## 🔧 Finetuning

We use Meta's [llama-recipes](https://github.com/meta-llama/llama-cookbook) for model fine-tuning.

### Step 1: Clone This Repo

<pre lang="markdown"> ```bash git clone https://github.com/ZixuanLiu4869/LLaVA-Sco.git cd LLaVA-Sco ``` </pre>


### Step 2: Download Base Dataset

Download the images from [LLaVA-CoT-100k](https://github.com/PKU-YuanGroup/LLaVA-CoT).

Update train/datasets/cot_dataset.py:


data_path = "PATH/TO/YOUR_JSONL"
image_base_path = "PATH/TO/IMAGE_FOLDER"


### Step 3: Install Dependencies

pip install llama-recipes

### Step 4: Run Finetuning

cd train

torchrun --nnodes 1 --nproc_per_node 2 finetuning.py \
  --enable_fsdp \
  --lr 1e-5 \
  --num_epochs 3 \
  --batch_size_training 4 \
  --model_name Xkev/Llama-3.2V-11B-cot \
  --dist_checkpoint_root_folder ./finetuned_model \
  --dist_checkpoint_folder fine-tuned \
  --use_fast_kernels \
  --dataset "custom_dataset" \
  --custom_dataset.test_split "test" \
  --custom_dataset.file "datasets/cot_dataset.py" \
  --run_validation False \
  --batching_strategy padding \
  --use_peft \
  --peft_method lora
  
## 🧪 Inference

python inference.py --finetuning_path PATH/TO/LORA/WEIGHTS


## 📌 TODO & Future Work

We envision LLaVA-SCo as a step toward making VLMs more interpretable and robust. Key directions for future improvements:

1. Scalability & Generalization: Explore how the self-correction mechanism scales to more tasks and domains.

2. Benchmark Analysis: Identify characteristics that influence when and why self-correction helps (or hurts) performance.

3. Self-Correction Placement: Conduct ablation studies on different positions and frequencies (would three turns further improve results?) of the correction stage.

4. Error Analysis: Categorize error types that are best mitigated by the self-correction mechanism.

5. Efficiency Comparison: Benchmark the compute cost against RL-based methods to validate the practical trade-offs.

   
## 📫 Citation (coming soon)

If you find this project useful or use it in your work, please consider citing our paper (link to be added after publication).

