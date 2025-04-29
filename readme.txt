🚀 Getting Started
🔧 Requirements
Install dependencies using conda or pip:

bash
Copy
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib scikit-learn pandas
Ensure access to a CUDA-enabled GPU for training.

🏋️‍♀️ How to Run
Data Preprocessing:

Run explore_data_and_preprocess.ipynb
➤ Cleans reports, filters redacted entries, validates images.

Split Data:

splitting_data.ipynb
➤ Generates train, val, and test folders.

Zero-Shot CLIP:

baseline_model.ipynb
➤ Tests pretrained CLIP on chest X-rays without fine-tuning.

Fine-Tuning CLIP:

sampletestCLIP.ipynb
➤ Loads CLIPChestXrayDataset, trains with contrastive loss, and saves the model.

t-SNE Visualization:

data_visualization.ipynb
➤ Generates 2D plot of learned image features (normal vs. abnormal).

Manual Inspection:

manual_predictions_vs_groundtruth.txt
➤ Shows example predictions from test set for visual validation.

🧪 Evaluation Metrics

Metric	Description
Top-1 Accuracy	How often the top retrieved report is correct
Cosine Score	Average similarity between image-text pairs
t-SNE Clustering	2D visualization of latent feature space
📊 Results

Phase	Top-1 Accuracy	Avg Cosine Similarity
Zero-shot CLIP	Low (~12%)	~0.22
Fine-tuned CLIP	↑ +41%	~0.30–0.33
🔍 Dataset
📚 Indiana University Chest X-ray Collection

Paired XML radiology reports + PNG images

7,470 images | 3,927 reports | 7,430 valid pairs used

Source: Open-i Dataset

🛠 Model Info
Vision Backbone: CLIP ViT-B/32

Text Encoder: CLIP BPE tokenizer

Contrastive Learning Loss

Trained for 5 epochs, batch size 32

📈 Key Features
✅ Zero-shot + fine-tuned CLIP comparison

✅ Real-world radiology text parsing (IMPRESSION/FINDINGS)

✅ Manual evaluation + Top-k accuracy

✅ No handcrafted labels or disease tags

✅ Visual clustering with t-SNE

🧠 Authors
Janardhan Reddy Guntaka – Concept + Data + Report

Sohith Sai Malyala – Code + Preprocessing + Visuals

Ram Prakash Yallavula – Validation + Final Review + Docs

📜 References
All academic references are included in the final LaTeX report.