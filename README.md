# OLMo Learns Chemistry

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-ChemOlmo--7b-blue)](https://huggingface.co/Codemaster67/ChemOlmo-7b)

## Can a General LLM Learn Chemistry?

In this project, I test if a general large language model (LLM) can learn to handle chemistry tasks well.

I start with **OLMo-7B**, a general LLM pre-trained on the [DOLMA](https://arxiv.org/abs/2402.00159) dataset. I then do continued pre-training using **QLoRA**.

I test the model on the **MoleculeNet** benchmark. I skip big datasets like QM9 and SIDER to save time and resources.

## Continued Pre-Training

I train the model on 2.1 million raw SMILES strings from the [smiles-molecules-chembl](https://huggingface.co/datasets/antoinebcx/smiles-molecules-chembl) dataset.

- **Training Script**: [RawSmiles.py](Olmo_learns_chemistry/Chembl_2M_and_instruction/RawSmiles.py) (for ChemOlmo-7B)

All benchmark code is in the [Notebooks folder](Notebooks/).

## Results

### Classification
![Classification Results](https://github.com/user-attachments/assets/3ab3eb65-3076-4d1f-bd5f-7fbb823e623e)

### Regression
![Regression Results](https://github.com/user-attachments/assets/eebaade0-6bf5-4000-a0b1-872d7788f8f2)

## References

1. [Dolma](https://arxiv.org/abs/2402.00159)
2. [OLMo-7B](https://huggingface.co/allenai/OLMo-7B)
3. [ZINC20](https://huggingface.co/datasets/zpn/zinc20)
4. [USPTO](https://huggingface.co/datasets/OpenMol/USPTO_1k_TPL-SFT)
5. [DeepChem](https://github.com/deepchem/deepchem)
6. [ChemBERTa-3](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-4glrl-v2)
7. [MoleculeNet](https://arxiv.org/abs/1703.00564)
