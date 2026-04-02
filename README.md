# Medical Visual Question Answering with RAG

A multimodal AI system that answers Yes/No questions about 
radiology images, powered by retrieval-augmented generation.

## Demo
![Demo Screenshot](demo.png)

## What it does
- Upload a radiology image
- Ask a clinical Yes/No question
- Get an AI prediction with confidence score
- See retrieved similar cases from knowledge base
- Get a grounded explanation with conflict detection

## Architecture
- Image Encoder  : ResNet50 (ImageNet pretrained, layer4 fine-tuned)
- Text Encoder   : BioClinicalBERT (MIMIC clinical notes pretrained)
- Fusion         : Text-guided attention gate
- Retrieval      : FAISS + SentenceTransformers (all-MiniLM-L6-v2)
- Demo           : Gradio

## RAG Pipeline
1. Question → SentenceTransformer → 384 dim vector
2. FAISS searches training knowledge base
3. Top 3 similar cases retrieved by cosine similarity
4. Retrieved evidence analyzed for Yes/No trend
5. Grounded explanation generated with conflict detection

## Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 75.00% |
| ROC AUC   | 0.82   |
| F1 Score  | 0.75   |
| Precision | 0.76   |
| Recall    | 0.74   |

## Dataset
VQA-RAD — radiology visual question answering dataset  
315 unique images, filtered to closed Yes/No questions

## Training Details
- Optimizer  : AdamW with differential learning rates
- Loss       : CrossEntropyLoss with label smoothing (0.1)
- Scheduler  : CosineAnnealingLR
- Epochs     : Early stopping at epoch 7 (best at epoch 3)
- Platform   : Kaggle (GPU T4)

## How to run
1. Clone this repo
2. Open the notebook on Kaggle
3. Add VQA-RAD dataset
4. Run all cells in order
5. Gradio demo launches at the end

## Tech Stack
- PyTorch
- HuggingFace Transformers
- FAISS
- SentenceTransformers
- Gradio
- Scikit-learn
