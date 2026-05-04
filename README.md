# 🧠 LSTM Context Window Impact Study

**Generative AI Lab — OEL Project**  
**Team:** Saad Ul Hassan, Abdul Rafay  
**Course:** Generative AI Lab  

---

## 📌 Overview

A systematic experimental study investigating how **sequence length (context window size)** impacts LSTM neural network performance across:

- 📉 Prediction accuracy (Cross-Entropy Loss & Perplexity)
- 🕐 Training time per epoch
- 💾 GPU memory consumption  
- 🧠 Context retention score (long-range dependency resolution)

---

## 🗂️ Project Structure

```
├── LSTM_Context_Window_Study.ipynb   # Main Colab notebook (run this first)
├── streamlit_app.py                  # Interactive Streamlit dashboard
├── requirements.txt                  # Python dependencies
├── results/                          # Auto-generated after running notebook
│   ├── all_results.json
│   ├── summary_table.csv
│   ├── dashboard.png
│   ├── heatmap_C*.png
│   └── vocab.pkl
└── checkpoints/                      # Saved model weights
    ├── model_C1.pt
    ├── model_C2.pt
    ├── model_C3.pt
    └── model_C4.pt
```

---

## 🔬 Experimental Configurations

| Config | Sequence Length | Description |
|--------|----------------|-------------|
| C1 | 15 | Short — baseline, limited context |
| C2 | 75 | Medium — balanced trade-off |
| C3 | 300 | Long — rich context, possible gradient issues |
| C4 | 600 | Very Long — stress test |

**Fixed Architecture:**
- 2-layer LSTM, hidden_size=256, embed_dim=128, dropout=0.3
- Dataset: Penn Treebank (language modeling)
- Optimizer: Adam (lr=0.001) with ReduceLROnPlateau
- Epochs: 10 (increase for better results)

---

## 🚀 How to Run

### Step 1 — Run Colab Notebook
1. Upload `LSTM_Context_Window_Study.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU (T4)**
3. Run all cells
4. Download `lstm_results.zip` and `lstm_checkpoints.zip`
5. Extract both zips into this project folder

### Step 2 — Run Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Step 3 — Push to GitHub
```bash
git init
git add LSTM_Context_Window_Study.ipynb streamlit_app.py requirements.txt README.md
git commit -m "Add LSTM Context Window Study — OEL Project"
git remote add origin https://github.com/YOUR_USERNAME/lstm-context-window-study
git push -u origin main
```

> **Note:** Add `results/` and `checkpoints/` to `.gitignore` or include them if you want to share pre-trained models.

---

## 📊 Results Dashboard (Streamlit)

The Streamlit app includes:
- **📊 Dashboard** — Overview KPIs + all charts
- **📈 Learning Curves** — Per-epoch training & validation loss
- **🔥 Activation Heatmaps** — Hidden state visualizations
- **🔮 Text Generation** — Live text generation from each config
- **📋 Summary & Recommendations** — Final analysis + CSV download

---

## 📝 Key Findings

1. **Accuracy improves** up to an optimal sequence length, beyond which returns diminish
2. **Memory scales super-linearly** with sequence length — C4 uses ~12× more GPU memory than C1
3. **Context retention peaks** at the medium-long range before gradient degradation sets in
4. **C2 (Medium)** offers the best accuracy-to-cost ratio for production NLP applications

---

## 📚 References

- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
- Marcus et al. (1993). *Penn Treebank.* Computational Linguistics.
- Melis et al. (2018). *On the State of the Art of Evaluation in Neural Language Models.* ICLR.
