"""
LSTM Context Window Impact Study — Streamlit Dashboard
Generative AI Lab | OEL Project
Team: Saad Ul Hassan, Abdul Rafay
"""

import streamlit as st
import json
import os
import math
import pickle

# ── Safe imports ──────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    st.error("Missing: numpy. Add `numpy==1.26.4` to requirements.txt"); st.stop()

try:
    import pandas as pd
except ImportError:
    st.error("Missing: pandas. Add `pandas==2.2.1` to requirements.txt"); st.stop()

try:
    import matplotlib
    matplotlib.use("Agg")  # required for Streamlit Cloud (no display)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    st.error("Missing: matplotlib. Add `matplotlib==3.8.3` to requirements.txt"); st.stop()

try:
    import torch
    import torch.nn as nn
except ImportError:
    st.error("Missing: torch. Add `torch==2.2.0` to requirements.txt"); st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Context Window Study",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main-header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    margin: 0;
}
.main-header p {
    color: #8b949e;
    margin: 0.3rem 0 0;
    font-size: 0.95rem;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-card .label {
    color: #8b949e;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.config-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.stSelectbox label, .stSlider label { color: #8b949e !important; }

div[data-testid="stMetricValue"] { color: #58a6ff !important; }

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Definition (must match notebook)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        return self.fc(out), hidden

# ─────────────────────────────────────────────────────────────────────────────
# Load Results
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {'C1': '#58a6ff', 'C2': '#3fb950', 'C3': '#f78166', 'C4': '#d2a8ff'}

@st.cache_data
def load_results():
    path = 'results/all_results.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_resource
def load_vocab():
    path = 'results/vocab.pkl'
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 LSTM Study")
    st.markdown("**Generative AI Lab — OEL**")
    st.markdown("---")
    st.markdown("**Team**")
    st.markdown("• Saad Ul Hassan\n• Abdul Rafay")
    st.markdown("---")
    st.markdown("**Configurations**")
    st.markdown("🔵 **C1** — Short (10–20)\n\n🟢 **C2** — Medium (50–100)\n\n🔴 **C3** — Long (200–500)\n\n🟣 **C4** — Very Long (500+)")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Dashboard",
        "📈 Learning Curves",
        "🔥 Activation Heatmaps",
        "🔮 Text Generation",
        "📋 Summary & Recommendations"
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧠 Context Window Impact Study — LSTM</h1>
  <p>Investigating how sequence length affects LSTM performance | Generative AI Lab OEL</p>
</div>
""", unsafe_allow_html=True)

results = load_results()

# ─────────────────────────────────────────────────────────────────────────────
# No Results Yet — Placeholder
# ─────────────────────────────────────────────────────────────────────────────
if results is None:
    st.warning("⚠️ No results found. Run the Colab notebook first, then place `results/` folder here.")
    st.info("**Steps:**\n1. Run `LSTM_Context_Window_Study.ipynb` in Colab\n2. Download `lstm_results.zip`\n3. Extract into the same folder as this `streamlit_app.py`\n4. Restart the app")

    # Show demo / placeholder charts
    st.markdown('<div class="section-title">📊 Preview (Demo Data)</div>', unsafe_allow_html=True)
    demo = {
        'C1': {'seq_len': 15,  'test_ppl': 245.3, 'retention': 0.031, 'mem': 512,  'time': 18},
        'C2': {'seq_len': 75,  'test_ppl': 178.1, 'retention': 0.047, 'mem': 1240, 'time': 42},
        'C3': {'seq_len': 300, 'test_ppl': 160.7, 'retention': 0.053, 'mem': 3200, 'time': 145},
        'C4': {'seq_len': 600, 'test_ppl': 172.4, 'retention': 0.049, 'mem': 6100, 'time': 310},
    }

    cols = st.columns(4)
    for i, (k, v) in enumerate(demo.items()):
        with cols[i]:
            st.metric(f"{k} — seq={v['seq_len']}", f"PPL: {v['test_ppl']}", f"Mem: {v['mem']}MB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')

    seq_lens = [v['seq_len'] for v in demo.values()]
    ppls     = [v['test_ppl'] for v in demo.values()]
    rets     = [v['retention'] for v in demo.values()]
    mems     = [v['mem'] for v in demo.values()]

    axes[0].plot(seq_lens, ppls, 'o-', color='#58a6ff', lw=2); axes[0].set_title('Perplexity', color='#e6edf3'); axes[0].set_facecolor('#161b22')
    axes[1].plot(seq_lens, rets, 's-', color='#f78166', lw=2); axes[1].set_title('Retention', color='#e6edf3');   axes[1].set_facecolor('#161b22')
    axes[2].plot(seq_lens, mems, '^-', color='#3fb950', lw=2); axes[2].set_title('GPU Memory', color='#e6edf3'); axes[2].set_facecolor('#161b22')
    for ax in axes:
        ax.tick_params(colors='#8b949e'); ax.set_xlabel('Seq Length', color='#8b949e')
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.grid(color='#21262d')
    fig.patch.set_facecolor('#0d1117')
    st.pyplot(fig)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Build Summary DataFrame
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for lbl, res in results.items():
    rows.append({
        'Label'     : lbl,
        'Config'    : res['config']['name'],
        'Seq Length': res['config']['seq_len'],
        'Test Loss' : round(res['test_loss'], 4),
        'Test PPL'  : round(res['test_ppl'], 2),
        'Retention' : round(res['retention_score'], 4),
        'Time/Epoch': round(res['avg_epoch_time'], 1),
        'GPU Mem MB': round(res['avg_gpu_memory'], 1),
    })
df = pd.DataFrame(rows)

LABELS   = df['Config'].tolist()
SEQ_LENS = df['Seq Length'].tolist()
COLORS   = [PALETTE.get(l, '#58a6ff') for l in df['Label'].tolist()]

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3', 'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'grid.color': '#21262d', 'font.family': 'monospace', 'font.size': 10,
})

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    # KPI cards
    best_ppl_row = df.loc[df['Test PPL'].idxmin()]
    best_ret_row = df.loc[df['Retention'].idxmax()]

    cols = st.columns(4)
    with cols[0]: st.metric("Best Test PPL", f"{best_ppl_row['Test PPL']:.1f}", best_ppl_row['Config'])
    with cols[1]: st.metric("Best Retention", f"{best_ret_row['Retention']:.3f}", best_ret_row['Config'])
    with cols[2]: st.metric("Configs Tested", "4", "C1 → C4")
    with cols[3]: st.metric("Dataset", "Penn Treebank", f"Vocab {10000:,}")

    st.markdown('<div class="section-title">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    if os.path.exists('results/dashboard.png'):
        st.image('results/dashboard.png', use_container_width=True)
    else:
        # Build inline
        fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

        # Perplexity bar
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(df['Label'], df['Test PPL'], color=COLORS, edgecolor='#30363d')
        for b, v in zip(bars, df['Test PPL']): ax1.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}', ha='center', fontsize=8)
        ax1.set_title('Test Perplexity', fontweight='bold'); ax1.grid(axis='y')

        # GPU Memory
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(df['Label'], df['GPU Mem MB'], color=COLORS, edgecolor='#30363d')
        for b, v in zip(bars, df['GPU Mem MB']): ax2.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}', ha='center', fontsize=8)
        ax2.set_title('GPU Memory (MB)', fontweight='bold'); ax2.grid(axis='y')

        # Training Time
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(df['Label'], df['Time/Epoch'], color=COLORS, edgecolor='#30363d')
        for b, v in zip(bars, df['Time/Epoch']): ax3.text(b.get_x()+b.get_width()/2, v+0.2, f'{v:.1f}s', ha='center', fontsize=8)
        ax3.set_title('Avg Time / Epoch', fontweight='bold'); ax3.grid(axis='y')

        # PPL vs SeqLen
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(SEQ_LENS, df['Test PPL'], 'o-', color='#58a6ff', lw=2.2, markersize=8)
        ax4.set_title('Perplexity vs Seq Length', fontweight='bold'); ax4.grid(True); ax4.set_xlabel('Sequence Length')

        # Retention vs SeqLen
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(SEQ_LENS, df['Retention'], 'D-', color='#f78166', lw=2.2, markersize=8)
        ax5.set_title('Retention vs Seq Length', fontweight='bold'); ax5.grid(True); ax5.set_xlabel('Sequence Length')

        # Memory vs SeqLen
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(SEQ_LENS, df['GPU Mem MB'], 's-', color='#3fb950', lw=2.2, markersize=8)
        ax6.set_title('Memory vs Seq Length', fontweight='bold'); ax6.grid(True); ax6.set_xlabel('Sequence Length')

        for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
            for sp in ax.spines.values(): sp.set_color('#30363d')
        st.pyplot(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Learning Curves
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Learning Curves":
    st.markdown('<div class="section-title">📈 Training & Validation Loss Curves</div>', unsafe_allow_html=True)

    metric = st.selectbox("Metric", ["valid_loss", "train_loss", "valid_ppl", "epoch_time", "gpu_memory_mb"])
    selected = st.multiselect("Show Configs", list(results.keys()), default=list(results.keys()))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    for lbl in selected:
        res = results[lbl]
        vals = res['history'].get(metric, [])
        epochs = range(1, len(vals)+1)
        ax.plot(epochs, vals, 'o-', color=PALETTE.get(lbl, '#58a6ff'),
                lw=2.2, markersize=5, label=res['config']['name'])
    ax.set_title(f'{metric.replace("_"," ").title()} per Epoch', fontweight='bold', fontsize=13)
    ax.set_xlabel('Epoch'); ax.grid(True); ax.legend()
    for sp in ax.spines.values(): sp.set_color('#30363d')
    st.pyplot(fig)

    # Per-epoch data table
    st.markdown('<div class="section-title">📋 Per-Epoch Data</div>', unsafe_allow_html=True)
    for lbl, res in results.items():
        if lbl in selected:
            h = res['history']
            edf = pd.DataFrame({
                'Epoch'   : range(1, len(h['train_loss'])+1),
                'Train Loss': [round(v,4) for v in h['train_loss']],
                'Valid Loss': [round(v,4) for v in h['valid_loss']],
                'Valid PPL' : [round(v,2) for v in h['valid_ppl']],
                'Time (s)'  : [round(v,1) for v in h['epoch_time']],
                'GPU MB'    : [round(v,1) for v in h['gpu_memory_mb']],
            })
            with st.expander(f"📂 {res['config']['name']}"):
                st.dataframe(edf, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Heatmaps
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔥 Activation Heatmaps":
    st.markdown('<div class="section-title">🔥 Hidden State Activation Heatmaps</div>', unsafe_allow_html=True)
    st.markdown("Visualize how the LSTM's hidden units activate across token positions for each configuration.")

    hmap_files = {
        'C1': 'results/heatmap_C1.png',
        'C2': 'results/heatmap_C2.png',
        'C3': 'results/heatmap_C3.png',
        'C4': 'results/heatmap_C4.png',
    }

    for lbl, path in hmap_files.items():
        res = results.get(lbl)
        if res and os.path.exists(path):
            st.markdown(f"**{res['config']['name']}** (seq_len={res['config']['seq_len']})")
            st.image(path, use_container_width=True)
        elif res:
            st.info(f"Heatmap for {res['config']['name']} not found. Re-run notebook section 8.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Text Generation
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Text Generation":
    st.markdown('<div class="section-title">🔮 Text Generation Demo</div>', unsafe_allow_html=True)
    vocab = load_vocab()

    if vocab is None:
        st.warning("Vocab file not found. Run the Colab notebook first.")
    else:
        cfg_choice = st.selectbox("Select Config", list(results.keys()),
                                  format_func=lambda x: results[x]['config']['name'])
        prompt = st.text_input("Prompt (space-separated words)", value="the stock market")
        n_words = st.slider("Words to generate", 10, 100, 30)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, step=0.1)

        if st.button("🚀 Generate"):
            ckpt = f"checkpoints/model_{cfg_choice}.pt"
            if not os.path.exists(ckpt):
                st.error(f"Checkpoint not found: {ckpt}")
            else:
                model = LSTMLanguageModel(vocab.vocab_size)
                model.load_state_dict(torch.load(ckpt, map_location='cpu'))
                model.eval()

                tokens = prompt.lower().split()
                ids = [vocab.word2idx.get(t, 1) for t in tokens]
                inp = torch.tensor([ids], dtype=torch.long)
                generated = list(tokens)

                with torch.no_grad():
                    hidden = None
                    for _ in range(n_words):
                        logits, hidden = model(inp, hidden)
                        logits = logits[0, -1, :] / temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, 1).item()
                        next_word = vocab.idx2word.get(next_id, '<unk>')
                        generated.append(next_word)
                        inp = torch.tensor([[next_id]], dtype=torch.long)

                output = ' '.join(generated).replace('<eos>', '\n')
                st.markdown("**Generated Text:**")
                st.code(output, language='text')
                st.caption(f"Config: {results[cfg_choice]['config']['name']} | "
                           f"seq_len={results[cfg_choice]['config']['seq_len']} | "
                           f"temp={temperature}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Summary & Recommendations
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋 Summary & Recommendations":
    st.markdown('<div class="section-title">📋 Results Summary</div>', unsafe_allow_html=True)

    st.dataframe(
        df[['Config','Seq Length','Test Loss','Test PPL','Retention','Time/Epoch','GPU Mem MB']],
        use_container_width=True
    )

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "lstm_results.csv", "text/csv")

    st.markdown('<div class="section-title">📝 Key Findings & Recommendations</div>', unsafe_allow_html=True)

    best_ppl = df.loc[df['Test PPL'].idxmin()]
    best_ret = df.loc[df['Retention'].idxmax()]

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**✅ Best Perplexity:** {best_ppl['Config']} (PPL = {best_ppl['Test PPL']})\n\n"
                   f"Sequence length {best_ppl['Seq Length']} achieved the lowest perplexity, "
                   f"indicating strongest language modeling performance.")
        st.info(f"**🧠 Best Context Retention:** {best_ret['Config']} (Score = {best_ret['Retention']})\n\n"
                f"Longer sequences enable the model to resolve long-range dependencies more effectively.")

    with col2:
        st.warning("**⚠️ Memory Trade-off:**\n\nGPU memory scales super-linearly with sequence length. "
                   "C4 (Very Long) may be impractical without high-end hardware.")
        st.error("**📉 Diminishing Returns:**\n\nBeyond the optimal context window, perplexity increases "
                 "due to gradient issues and overfitting on noisy long-range patterns.")

    st.markdown("### 🎯 Production Recommendation")
    st.markdown("""
| Use Case | Recommended Config | Reason |
|---|---|---|
| Real-time inference | **C1 — Short** | Fast, memory-efficient |
| General NLP tasks | **C2 — Medium** | Best accuracy-to-cost ratio |
| Document-level tasks | **C3 — Long** | Rich context, manageable cost |
| Research / high-end | **C4 — Very Long** | Maximum context, needs large GPU |
""")

    st.markdown("### 📚 References")
    st.markdown("""
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
- Marcus et al. (1993). *Penn Treebank.* Computational Linguistics.
- Melis et al. (2018). *On the State of the Art of Evaluation in Neural Language Models.* ICLR.
- Bengio et al. (1994). *Learning Long-term Dependencies with Gradient Descent is Difficult.*
""")
