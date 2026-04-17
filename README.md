# Fine Tuning a VLM for Bangla Meme Classification

This project fine tunes a Vision Language Model (VLM) to classify Bangla memes into four target categories. The model learns to look at a meme image and predict which group the meme is aimed at.

## What This Project Does

The goal is simple. Given a meme image, the model reads the image and replies with one label from this list:

- **Genders**
- **Neutral**
- **Politics**
- **Religion**

I take a pretrained vision language model and teach it to do this task using a small Bangla meme dataset. I use QLoRA so that fine tuning stays light and can run on a single GPU.

## Dataset

I use the Bangla meme dataset introduced in the paper below. The dataset has meme images paired with target labels, and it was built to study target aware aggression in memes.

> Ahsan, S., Hossain, E., Sharif, O., Das, A., Hoque, M. M., and Dewan, M. *A Multimodal Framework to Detect Target Aware Aggression in Memes.* In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2487-2500, 2024.

```
@inproceedings{ahsan2024multimodal,
  title={A Multimodal Framework to Detect Target Aware Aggression in Memes},
  author={Ahsan, Shawly and Hossain, Eftekhar and Sharif, Omar and Das, Avishek and Hoque, Mohammed Moshiul and Dewan, M},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2487--2500},
  year={2024}
}
```

The raw data lives under [Train/](Train/) and contains an image folder plus a CSV file with the `Image_name` and `Target` columns.

**Note on captions.** The original dataset also ships a text caption for each training meme. In this project I ignore that column on purpose and train the VLM with only image and label pairs. The idea is to let the vision language model read the meme directly from pixels, without leaning on a precomputed caption as a shortcut.

## Model

I fine tune **Qwen2-VL-2B-Instruct** using the Unsloth library. The base model is loaded in 4 bit to save memory, and I attach LoRA adapters on top. Only the language side is trained. The vision encoder stays frozen.

Main model settings:

| Setting | Value |
| --- | --- |
| Base model | `unsloth/Qwen2-VL-2B-Instruct` |
| Load in 4 bit | Yes |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Max sequence length | 1024 |

### Why Unsloth

I picked Unsloth instead of plain Hugging Face Transformers for a few practical reasons:

- **Memory friendly.** Unsloth fuses its own kernels for attention and MLP, and pairs that with 4 bit loading through bitsandbytes. A 2B parameter VLM that would normally need around 12 to 16 GB of VRAM in fp16 fits comfortably inside a single free tier Kaggle T4 (16 GB) with room left for batch and activations.
- **Faster training.** The custom Triton kernels and gradient checkpointing mode (`use_gradient_checkpointing="unsloth"`) cut wall clock time by roughly 2x compared to a vanilla PEFT + Transformers setup on the same hardware. For a small project like this, that turns an overnight run into an afternoon run.
- **Clean VLM support.** `FastVisionModel` hides a lot of the rough edges around vision language models. Things like the image data collator, switching the model between training and inference modes, and saving a merged 16 bit model work out of the box with one or two lines.
- **Same ecosystem.** Under the hood it still returns a regular Hugging Face model and tokenizer, so the TRL `SFTTrainer`, PEFT adapters, and standard checkpoints all keep working. Nothing is locked in.

In short, Unsloth is the cheapest path to a working QLoRA fine tune on a small GPU budget, without giving up the Hugging Face tooling around it.

## Data Preprocessing

Before training, I clean the images using [data_preprocessing.py](data_preprocessing.py). This step does a few simple things:

1. Drops any image that is fully grayscale, since colour information is a useful signal for memes.
2. Resizes each image so the longest side is at most 1024 pixels.
3. Rounds the width and height so both are divisible by 28. Qwen2-VL prefers image sizes that fit its patch grid, and this avoids padding waste.
4. Saves the cleaned images into `Train/Image_processed/` and writes a matching `Train/Train_processed.csv`.

Run it with:

```bash
python data_preprocessing.py
```

## Training

The training code is in [main.ipynb](main.ipynb). The notebook is written to run on Kaggle, but you can adapt the paths in the `CONFIG` dictionary for any machine with a GPU.

Steps in the notebook:

1. Load the Qwen2-VL model in 4 bit through Unsloth.
2. Read the preprocessed CSV and split the data into train, validation, and test sets. The split is 80 / 10 / 10 and is stratified by the target label so every class is represented fairly.
3. Turn each row into a chat style conversation. The user turn has the image plus a short instruction that lists the allowed labels. The assistant turn is just the target label.
4. Attach LoRA adapters and train with the TRL `SFTTrainer`.
5. Save the adapter, and optionally a merged 16 bit full model, to the output directory.

Key training settings:

| Setting | Value |
| --- | --- |
| Per device batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Max grad norm | 0.3 |
| Optimiser | paged_adamw_8bit |
| Epochs | 7 (total) |
| Eval strategy | every epoch |
| Save strategy | every 100 steps |

### Resuming From a Checkpoint

The `CONFIG["resume_from_checkpoint"]` field lets you continue training from a saved step. The folder [checkpoint-578/](checkpoint-578/) in this repo is the checkpoint at the end of epoch 2. To keep going from there, just point `resume_from_checkpoint` at it. If the path does not exist, training starts fresh.

Note that `num_train_epochs` is the total number of epochs, not the extra number. So to train two more epochs on top of checkpoint-578, set `num_train_epochs` to 4.

## Evaluation

After training, the notebook runs the model on the held out test split. For each meme, it generates a short reply, picks out the label name that appears in the reply, and compares it to the ground truth.

The evaluation saves two files:

- `eval/test_predictions.csv` with one row per test image.
- `eval/test_metrics.json` with overall and per class accuracy.

## Results

I trained three separate runs for 2, 3, and 7 epochs and evaluated each on the same 290 meme test split. The full metric files live under [Results/](Results/):

- [Results/epoch-2-results/test_metrics.json](Results/epoch-2-results/test_metrics.json)
- [Results/epoch-3-results/test_metrics.json](Results/epoch-3-results/test_metrics.json)
- [Results/epoch-7-results/test_metrics.json](Results/epoch-7-results/test_metrics.json)

### Comparative Analysis

All three runs land on the exact same overall accuracy, but they get there by balancing the classes differently.

| Metric | 2 epochs | 3 epochs | 7 epochs |
| --- | --- | --- | --- |
| Overall accuracy | **82.07 %** (238 / 290) | **82.07 %** (238 / 290) | **82.07 %** (238 / 290) |
| Macro average accuracy | **83.24 %** | 82.39 % | 82.77 % |
| Religion | **98.41 %** (62 / 63) | 96.83 % (61 / 63) | 90.48 % (57 / 63) |
| Politics | **86.67 %** (52 / 60) | 80.00 % (48 / 60) | 85.00 % (51 / 60) |
| Neutral | 76.04 % (73 / 96) | **82.29 %** (79 / 96) | 78.13 % (75 / 96) |
| Genders | 71.83 % (51 / 71) | 70.42 % (50 / 71) | **77.46 %** (55 / 71) |

A few observations:

- **The global accuracy is flat at 82 percent across all three runs.** Extra epochs do not move the headline number, they only shift which classes win and which ones lose. The model is trading correct answers between buckets, not unlocking new capacity.
- **Epoch 2 has the best macro average (83.24 %).** It wins on the easy, high signal classes (Religion, Politics) and that lifts the mean of per class scores, which is the fairer metric on this imbalanced split.
- **Epoch 3 is the Neutral specialist.** Neutral jumps from 76 to 82 percent, but Politics drops 6.7 points and Religion slips by 1.6. The model is starting to favour "no target" as a default answer.
- **Epoch 7 is the Genders specialist.** It is the only run that cracks 77 percent on Genders, but it pays for that with a 7.9 point collapse on Religion. This is classic late stage drift: the model keeps searching for harder patterns and forgets the easy ones.
- **Religion degrades monotonically with more training (98 → 97 → 90).** Once a class is near saturated, further epochs hurt it. There is nothing left to learn, and the gradient from the harder classes pulls the decision boundary away from the Religion centroid.
- **Genders improves monotonically with more training (72 → 70 → 77).** This is the only class where longer training actually helps, because its cues are the most subtle and need more passes to be picked up.

### Why 2 Epochs is the Best Choice

Given the numbers above, the 2 epoch checkpoint is the one I recommend for deployment:

- **Best macro accuracy.** 83.24 percent macro beats both 3 and 7 epochs. On an imbalanced test split, macro is the honest score and epoch 2 wins it.
- **No regression on any class.** Compared to 3 epochs it is better on 3 out of 4 classes. Compared to 7 epochs it is better on Religion and Politics and only worse on Genders.
- **About 3.5x cheaper to train.** On the same GPU, 7 epochs costs roughly 3.5x the wall clock of 2 epochs, and 3 epochs costs 1.5x. Since the overall accuracy is identical, every extra epoch is paying more compute for the same answer count.
- **Lower overfitting risk.** Religion loses 8 points going from 2 to 7 epochs. That is a clear sign the model is starting to memorise harder examples at the cost of easier ones. Stopping early avoids that drift.
- **Simpler to reproduce.** Shorter runs are easier to rerun, easier to sweep across seeds, and cheaper to iterate on when you want to try unfreezing the vision layers or adding class weights.

In short, 2 epochs is the Pareto optimal point on this setup: it matches the best overall accuracy, has the best balanced accuracy, and is the fastest to train. More epochs would only be worth it if you specifically need the Genders class boosted and are willing to give up Religion and Politics to get it.

### Analytical Insights (Epoch 2)

A few things stand out when you look past the overall number for the recommended 2 epoch run:

- **The classes are imbalanced in the test split.** Neutral is the largest bucket with 96 samples, then Genders with 71, Politics with 60, and Religion with only 63. A flat overall accuracy hides this. The macro average accuracy (the mean of the four per class numbers) is about 83.24 percent, which is actually a touch higher than the micro average of 82.07 percent. That tells me the model is not just winning by memorising the biggest class.

- **Religion is almost saturated.** 62 out of 63 correct is 98.41 percent. Religion memes in Bangla often carry very distinct visual markers (symbols, robes, places of worship, iconography) and specific script cues. The VLM locks onto these fast, so even 2 epochs with LoRA are enough. There is very little room left to grow here, and the 7 epoch run confirms it: extra training drops Religion by 8 points.

- **Politics is next easiest.** 86.67 percent. Political memes lean on recognisable faces, party colours, and flags. The vision encoder of Qwen2-VL already knows many of these from pretraining, so the LoRA head only needs to learn the mapping from "this face or flag" to the Politics label.

- **Neutral is the hardest honest class.** 76.04 percent, and it is the largest bucket, so its errors pull the overall number down the most. Neutral is defined by the absence of a target, which is a much weaker signal than the presence of one. The model probably confuses Neutral with Genders or Politics whenever a meme contains a person, because "person in frame" is not enough to tell you whether the joke targets that person.

- **Genders is the weakest class at 71.83 percent.** This is the class with the most subtle cues. The target is often identified only by a Bangla text overlay or by context, not by a clear visual marker. Two things likely hurt here: (a) the vision encoder stays frozen, so any fine grained Bangla script reading has to happen through the frozen ViT tokens, and (b) the class is mid sized, so the model does not get as much signal as it does on Neutral.

- **Where the gains probably are.** The gap between Religion (98) and Genders (72) is 26 points. Closing that gap will not come from more epochs alone, the 7 epoch run already proved that. Unfreezing the vision layers (set `finetune_vision_layers=True`), adding class weighted loss, or feeding an OCR string as extra context would all target the Genders and Neutral failure modes more directly than another pass of plain LoRA.

## Project Layout

```
VLM-Finetuning/
├── README.md                 # this file
├── main.ipynb                # training and evaluation notebook
├── data_preprocessing.py     # image cleanup and CSV rewriting
├── Results/                  # test metrics and predictions for 2, 3, 7 epoch runs
├── checkpoint-578/           # LoRA checkpoint at end of epoch 2
└── Train/                    # raw dataset (images and CSV)
```

## Requirements

- Python 3.10 or newer
- A CUDA GPU with at least 12 GB of memory for 4 bit training
- `unsloth`, `transformers`, `trl`, `peft`, `bitsandbytes`, `pandas`, `scikit-learn`, `Pillow`, `tqdm`

On Kaggle or Colab you can simply run the first cell of the notebook:

```bash
pip install unsloth
```

which pulls the rest of the stack as dependencies.

## Credits

- Dataset and task design: Ahsan et al., EACL 2024 (citation above).
- Base model: Qwen2-VL-2B-Instruct by Alibaba.
- Training stack: Unsloth and TRL.
