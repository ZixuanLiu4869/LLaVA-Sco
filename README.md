LLaVA-SCo: Teach Vision Language Model to Self-Correct



Dataset

You can download the dataset from https://drive.google.com/file/d/1937sO0WJwKIAT1661wk7jCgw4-b_kAyO/view?usp=sharing


Finetuning

We are uing llama-recipes: https://github.com/meta-llama/llama-cookbook


git clone https://github.com/ZixuanLiu4869/LLaVA-Sco.git


First, download the image of LLaVA-CoT-100k from https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k


Next, change train/datasets/cot_dataset.py

line 57: data_path = "PATH/TO/DATA_JSONL_FILE"
line 58: image_base_path = "PATH/TO/IMAGE"


Next, 

pip install llama-recipes

cd train

torchrun --nnodes 1 --nproc_per_node 2  finetuning.py --enable_fsdp --lr 1e-5  --num_epochs 3 --batch_size_training 4 --model_name Xkev/Llama-3.2V-11B-cot --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "datasets/cot_dataset.py"  --run_validation False --batching_strategy padding  --use_peft --peft_method lora
