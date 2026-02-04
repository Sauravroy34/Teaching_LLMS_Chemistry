# Teaching LLMs Chemistry

## Can a general LLM learn chemistry?

In this project, I try to find out whether a general LLM can be taught to master a specific domain (chemistry in this example).

I took a general LLM — **OLMo-7B** — which was pre-trained on the **[DOLMA](https://arxiv.org/abs/2402.00159)** dataset, and performed continued pre-training using **QLoRA**.

The model was benchmarked on the **MoleculeNet** dataset (excluding some datasets like QM9 and SIDER due to their enormous size).

## Continued Pre-training

**First step:**  
The model was first pre-trained on a subset of the **[ZINC20](https://huggingface.co/datasets/zpn/zinc20)** dataset. The goal was to teach the model about SMILES representation. Due to compute constraints, only a **10k subset** could be used. The pre-training method was **QLoRA** on Kaggle with **2×T4 GPUs**. It took around **8 hours** of runtime.

**Second step:**  
The model was then continued pre-training on a subset of the **[USPTO](https://huggingface.co/datasets/OpenMol/USPTO_1k_TPL-SFT)** reaction dataset. The goal here was — since the model had already learned some chemical structure representation (through ZINC20 pre-training) — to infuse actual chemistry knowledge into the model's weights. Again, due to compute constraints, only a **10k subset** was used. The method was **QLoRA** on Kaggle with **2×T4 GPUs**. This took around **9 hours** of runtime.

I compared the model with other available chemistry-specific transformer models and a random forest baseline. After continued pre-training, the model's scores did improve — however, not enough to overtake some of the chemistry-specific transformer models.

More details about the pre-training can be found in the `Pre_training_notebooks` folder.

## Results

#### Classification
<img width="1374" height="907" alt="Classification" src="https://github.com/user-attachments/assets/8bbdb313-22e7-4b62-b83f-b3d6ca32802f" />

#### Regression
<img width="1380" height="689" alt="Regression" src="https://github.com/user-attachments/assets/49a7eb2a-68b8-43b1-98d6-7bc40ec699e8" />

## References

1. [Dolma](https://arxiv.org/abs/2402.00159)  
2. [OLMo-7B](https://huggingface.co/allenai/OLMo-7B)  
3. [ZINC20](https://huggingface.co/datasets/zpn/zinc20)  
4. [USPTO](https://huggingface.co/datasets/OpenMol/USPTO_1k_TPL-SFT)  
5. [DEEPCHEM](https://github.com/deepchem/deepchem)  
6. [ChemBERTa-3](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-4glrl-v2)
