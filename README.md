# Imperial College London MEng Individual Project Archive - VisualQA

This repository contains the code and experiments related to my MEng individual project at Imperial College London.

## Directory Structure

### Model Training / Eval

- `src/eval`: Main evaluation frameworks. Adaptors for each model can be found in `src/eval/adaptors`. Current supported models include our model VisualQA (obviously), XrayGPT, LLaVA-Med, and RadFM. Dataset support includes MIMIC-CXR and NIH-14. All configurations need to be changed in `src/eval/eval_config.yaml` before running evaluations to specify model type, checkpoint directory, dataset name, and dataset directory.
- `src/train`: Contains experiments related to CLIP, including SigCLIP, +Classifier, and +Divergence.
- `src/train_v2`: Contains training scripts for VisualQA using Hugging Face transformers, Trainer, and PEFT. This is the main focus of the project. Detailed information about the models used can be found in `src/train_v2/models`.

### Others

- `openai_script/`: Scripts for running OpenAI API for data curation and multi-turn dialogue.
- `local_test/`: Run tests with records of model outputs.
- `bleu/` and `rouge/`: Local packages of language evaluation scripts, please ignore.

