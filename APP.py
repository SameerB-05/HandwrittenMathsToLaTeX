import io
import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from torchvision import transforms
from tamer.lit_tamer import LitTAMER
from tamer.datamodule import vocab

# ── paths ─────vocab.init─────────────────────────────────────────────────────────────
V1_CKPT  = r"lightning_logs\version_1\checkpoints\epoch=51-step=162967-val_ExpRate=0.6851.ckpt"
V4_CKPT  = r"lightning_logs\version_4\checkpoints\finetune_epoch5_loss0.7993.ckpt"
DICT_V1  = r"data\hme100k\dictionary.txt"               # 248-token vocab
DICT_V4  = r"lightning_logs\version_4\dictionary.txt"   # 334-token vocab
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ── snapshot both word-lists before touching the singleton ─────────────────
def load_words(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

words_v1 = load_words(DICT_V1)   # len == 248
words_v4 = load_words(DICT_V4)   # len == 334

def set_vocab(words):
    """Overwrite the global vocab singleton's mappings in-place."""
    vocab.word2idx = {
        "<pad>": vocab.PAD_IDX,
        "<sos>": vocab.SOS_IDX,
        "<eos>": vocab.EOS_IDX,
    }
    for w in words:
        if w not in vocab.word2idx:
            vocab.word2idx[w] = len(vocab.word2idx)

    vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}
    if hasattr(vocab, 'words'):
        vocab.words = words

# Initialise singleton with v1 (248) so model_v1 loads cleanly
vocab.init(DICT_V1)

# ── model v1 ───────────────────────────────────────────────────────────────
print("Loading model v1  (vocab=248)...")
model_v1 = LitTAMER.load_from_checkpoint(V1_CKPT, map_location=DEVICE)
model_v1.eval()
model_v1.to(DEVICE)

# ── model v4 ───────────────────────────────────────────────────────────────
# Load base arch on CPU → patch vocab layers → load finetuned weights → GPU
print("Loading model v4  (vocab=334, finetuned)...")
model_v4 = LitTAMER.load_from_checkpoint(V1_CKPT, map_location="cpu")
model_v4.tamer_model.decoder.word_embed[0] = torch.nn.Embedding(334, 256)
model_v4.tamer_model.decoder.proj          = torch.nn.Linear(256, 334)

finetuned_weights = torch.load(V4_CKPT, map_location="cpu")
model_v4.tamer_model.load_state_dict(finetuned_weights, strict=False)

# manually load the vocab-expanded layers since strict=False skips shape mismatches
sd = finetuned_weights
model_v4.tamer_model.decoder.word_embed[0].weight.data = sd["decoder.word_embed.0.weight"]
model_v4.tamer_model.decoder.proj.weight.data          = sd["decoder.proj.weight"]
model_v4.tamer_model.decoder.proj.bias.data            = sd["decoder.proj.bias"]

model_v4.eval()
model_v4.to(DEVICE)

print(f"Both models ready on {DEVICE}!")

# ── shared image transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7931], std=[0.1738])
])

app = Flask(__name__)

# ── HTML ───────────────────────────────────────────────────────────────────
# ── HTML ───────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Handwritten Math to LaTeX — Model Comparison</title>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: Arial, sans-serif;
      max-width: 1000px;
      margin: 50px auto;
      padding: 20px;
      background: #f0f2f5;
    }

    h1 { color: #222; margin-bottom: 24px; font-size: 1.6rem; }
    h3 { color: #333; margin-bottom: 14px; font-size: 1rem; }

    .box {
      background: #fff;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
      margin-bottom: 20px;
    }

    input[type=file] { font-size: 15px; }

    button.primary {
      background: #ff6b00;
      color: #fff;
      border: none;
      padding: 12px 30px;
      font-size: 16px;
      border-radius: 6px;
      cursor: pointer;
    }

    button.primary:hover { background: #e55a00; }
    button.primary:disabled {
      background: #ccc;
      cursor: not-allowed;
    }

    #loading {
      display: none;
      color: #888;
      margin-top: 14px;
      font-size: 15px;
    }

    /* Upload row horizontal layout */
    .upload-row {
      display: flex;
      align-items: center;
      gap: 20px;
      flex-wrap: wrap;
    }

    .upload-controls {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    #preview {
      width: 180px;
      max-height: 120px;
      display: none;
      border: 1px solid #ddd;
      border-radius: 6px;
      object-fit: contain;
      background: #fff;
      padding: 4px;
    }

    .comparison {
      display: none;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    .comparison.visible { display: grid; }

    @media (max-width: 680px) {
      .comparison { grid-template-columns: 1fr; }

      .upload-row {
        flex-direction: column;
        align-items: flex-start;
      }
    }

    .model-box {
      background: #fff;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    .model-title {
      font-size: .8rem;
      text-transform: uppercase;
      letter-spacing: .5px;
      color: #888;
      margin-bottom: 14px;
    }

    .model-title strong {
      font-size: 1rem;
      color: #222;
      display: block;
      margin-bottom: 2px;
    }

    .label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .5px;
      color: #888;
      margin-bottom: 6px;
    }

    .latex-code {
      background: #1e1e1e;
      color: #00ff88;
      padding: 14px 16px;
      border-radius: 6px;
      font-family: monospace;
      font-size: 14px;
      min-height: 48px;
      word-break: break-all;
      line-height: 1.5;
    }

    .latex-code.error { color: #ff4444; }

    button.copy-btn {
      background: #444;
      color: #fff;
      border: none;
      font-size: 12px;
      padding: 5px 14px;
      border-radius: 5px;
      cursor: pointer;
      margin-top: 8px;
    }

    button.copy-btn:hover { background: #222; }

    .rendered-box {
      background: #fafafa;
      border: 1px solid #e0e0e0;
      padding: 22px 16px;
      border-radius: 6px;
      font-size: 20px;
      min-height: 60px;
      text-align: center;
      margin-top: 14px;
    }
  </style>
</head>

<body>

<h1>✏️ Handwritten Math → LaTeX · Model Comparison</h1>

<div class="box">
  <h3>Upload a handwritten math image</h3>

  <div class="upload-row">

    <div class="upload-controls">
      <input type="file" id="fileInput" accept="image/*" onchange="previewImage()">

      <button class="primary" id="convertBtn" onclick="convert()" disabled>
        Convert to LaTeX
      </button>
    </div>

    <img id="preview">

  </div>

  <div id="loading">
    ⏳ Running beam search on both models, please wait…
  </div>
</div>

<div class="comparison" id="comparison">

  <div class="model-box">
    <div class="model-title">
      <strong>Version 1</strong>
      vocab 248 · baseline
    </div>

    <div class="label">LaTeX Code</div>
    <div class="latex-code" id="v1-code"></div>

    <button class="copy-btn" onclick="copyLatex('v1-code', this)">
      Copy
    </button>

    <div class="label" style="margin-top:14px;">Rendered Expression</div>
    <div class="rendered-box" id="v1-rendered"></div>
  </div>

  <div class="model-box">
    <div class="model-title">
      <strong>Version 4</strong>
      vocab 334 · finetuned
    </div>

    <div class="label">LaTeX Code</div>
    <div class="latex-code" id="v4-code"></div>

    <button class="copy-btn" onclick="copyLatex('v4-code', this)">
      Copy
    </button>

    <div class="label" style="margin-top:14px;">Rendered Expression</div>
    <div class="rendered-box" id="v4-rendered"></div>
  </div>

</div>

<script>
function previewImage() {
  const file = document.getElementById('fileInput').files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = e => {
    const prev = document.getElementById('preview');
    prev.src = e.target.result;
    prev.style.display = 'block';

    document.getElementById('convertBtn').disabled = false;
  };

  reader.readAsDataURL(file);
}

function setLoading(on) {
  document.getElementById('loading').style.display = on ? 'block' : 'none';
  document.getElementById('convertBtn').disabled = on;

  if (on) {
    ['v1-code','v4-code'].forEach(id => {
      const el = document.getElementById(id);
      el.className = 'latex-code';
      el.textContent = 'Running…';
    });

    ['v1-rendered','v4-rendered'].forEach(id =>
      document.getElementById(id).innerHTML = ''
    );

    document.getElementById('comparison').classList.add('visible');
  }
}

function fillCard(prefix, data) {
  const codeEl = document.getElementById(prefix + '-code');
  const rendEl = document.getElementById(prefix + '-rendered');

  if (data.error) {
    codeEl.className = 'latex-code error';
    codeEl.textContent = 'Error: ' + data.error;
    rendEl.textContent = '';
  } else {
    const latex = data.latex;

    codeEl.className = 'latex-code';
    codeEl.textContent = latex;

    rendEl.innerHTML = '\\\\(' + latex + '\\\\)';

    MathJax.typesetPromise([rendEl]).catch(err => {
      rendEl.textContent = '(Could not render: ' + latex + ')';
    });
  }
}

function convert() {
  const file = document.getElementById('fileInput').files[0];

  if (!file) {
    alert('Please select an image first!');
    return;
  }

  setLoading(true);

  const fd = new FormData();
  fd.append('image', file);

  fetch('/predict', {
    method: 'POST',
    body: fd
  })
  .then(r => r.json())
  .then(data => {
    setLoading(false);
    fillCard('v1', data.v1);
    fillCard('v4', data.v4);
  })
  .catch(err => {
    setLoading(false);
    alert('Request failed: ' + err);
  });
}

function copyLatex(elId, btn) {
  const text = document.getElementById(elId).textContent.trim();

  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 2000);
  });
}
</script>

</body>
</html>
"""

# ── inference helpers ──────────────────────────────────────────────────────
def make_mask(img_tensor):
    return torch.zeros(
        1, img_tensor.shape[2], img_tensor.shape[3],
        dtype=torch.bool,
        device=img_tensor.device
    )


def run_inference(model, img_tensor, mask, words):
    """
    Swap the global vocab to `words` before running beam search so that:
      - model_v1 never sees indices >= 248
      - model_v4 can use the full 334-token range
    """
    set_vocab(words)
    with torch.no_grad():
        hyps = model.tamer_model.beam_search(img_tensor, mask, **model.hparams)
    return vocab.indices2label(hyps[0].seq)


# ── routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        file  = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert("L")

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        mask = make_mask(img_tensor)

        print(f"Image size: {image.size}, tensor: {img_tensor.shape}")

        # v1 — 248-token vocab (prevents OOB index assertion on GPU)
        print("Running v1 beam search  (vocab=248)...")
        try:
            latex_v1 = run_inference(model_v1, img_tensor, mask, words_v1)
            v1_result = {'latex': latex_v1}
        except Exception as e:
            v1_result = {'error': str(e)}

        # v4 — 334-token vocab
        print("Running v4 beam search  (vocab=334)...")
        try:
            latex_v4 = run_inference(model_v4, img_tensor, mask, words_v4)
            v4_result = {'latex': latex_v4}
        except Exception as e:
            v4_result = {'error': str(e)}

        print(f"v1: {v1_result}")
        print(f"v4: {v4_result}")

        return jsonify({'v1': v1_result, 'v4': v4_result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'v1': {'error': str(e)}, 'v4': {'error': str(e)}})


if __name__ == '__main__':
    app.run(debug=False, port=5000)