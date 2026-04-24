import io
import base64
import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from torchvision import transforms
from tamer.lit_tamer import LitTAMER
from tamer.datamodule import vocab

CKPT_PATH = r"lightning_logs\version_1\checkpoints\epoch=51-step=162967-val_ExpRate=0.6851.ckpt"
#DICT_PATH = r"data\hme100k\dictionary.txt"
DICT_PATH = r"lightning_logs\version_4\dictionary.txt"  # 334-token vocab
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading vocab...")
vocab.init(DICT_PATH)
print("Loading model...")
model = LitTAMER.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)

# Manually resize the vocab-dependent layers to 334
model.tamer_model.decoder.word_embed[0] = torch.nn.Embedding(334, 256)
model.tamer_model.decoder.proj = torch.nn.Linear(256, 334)

# then override with finetuned weights
finetuned_weights = torch.load(
    r"lightning_logs\version_4\checkpoints\finetune_epoch5_loss0.7993.ckpt",
    map_location=DEVICE
)
model.tamer_model.load_state_dict(finetuned_weights, strict=False)

# manually load the vocab-expanded layers since strict=False skips shape mismatches
sd = finetuned_weights
model.tamer_model.decoder.word_embed[0].weight.data = sd["decoder.word_embed.0.weight"]
model.tamer_model.decoder.proj.weight.data          = sd["decoder.proj.weight"]
model.tamer_model.decoder.proj.bias.data            = sd["decoder.proj.bias"]

model.eval()
model.to(DEVICE)
print(f"Model ready on {DEVICE}!")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7931], std=[0.1738])
])


app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Handwritten Math to LaTeX</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; max-width: 860px; margin: 50px auto; padding: 20px; background: #f0f2f5; }
        h1 { color: #222; margin-bottom: 24px; }
        .box { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin: 20px 0; }
        input[type=file] { margin: 10px 0; font-size: 15px; }
        button { background: #ff6b00; color: white; border: none; padding: 12px 30px; font-size: 16px; border-radius: 6px; cursor: pointer; margin-top: 10px; }
        button:hover { background: #e55a00; }
        #preview { max-width: 100%; max-height: 220px; margin: 12px 0; display: none; border: 1px solid #ddd; border-radius: 6px; }
        #loading { display: none; color: #888; margin-top: 12px; font-size: 15px; }
        .result-box { margin-top: 10px; }
        .label { font-size: 13px; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
        #latex-code { background: #1e1e1e; color: #00ff88; padding: 16px; border-radius: 6px; font-family: monospace; font-size: 16px; min-height: 48px; word-break: break-all; }
        #latex-rendered { background: #fafafa; border: 1px solid #e0e0e0; padding: 24px; border-radius: 6px; font-size: 22px; min-height: 60px; text-align: center; margin-top: 16px; }
        .hidden { display: none; }
        #copy-btn { background: #444; font-size: 13px; padding: 6px 16px; margin-top: 8px; }
        #copy-btn:hover { background: #222; }
    </style>
</head>
<body>
    <h1>✏️ Handwritten Math → LaTeX</h1>

    <div class="box">
        <h3>Upload a handwritten math image</h3>
        <input type="file" id="fileInput" accept="image/*" onchange="previewImage()">
        <br>
        <img id="preview">
        <br>
        <button onclick="convert()">Convert to LaTeX</button>
        <div id="loading">⏳ Running beam search, please wait...</div>
    </div>

    <div class="box hidden" id="output-box">
        <h3>Result</h3>
        <div class="result-box">
            <div class="label">LaTeX Code</div>
            <div id="latex-code"></div>
            <button id="copy-btn" onclick="copyLatex()">Copy</button>
        </div>
        <div class="result-box">
            <div class="label">Rendered Expression</div>
            <div id="latex-rendered"></div>
        </div>
    </div>

    <script>
        function previewImage() {
            const file = document.getElementById('fileInput').files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = e => {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        }

        function convert() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) { alert('Please select an image first!'); return; }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('output-box').classList.add('hidden');

            const formData = new FormData();
            formData.append('image', file);

            fetch('/predict', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('output-box').classList.remove('hidden');

                    if (data.error) {
                        document.getElementById('latex-code').style.color = '#ff4444';
                        document.getElementById('latex-code').textContent = 'Error: ' + data.error;
                        document.getElementById('latex-rendered').textContent = '';
                    } else {
                        const latex = data.latex;
                        document.getElementById('latex-code').style.color = '#00ff88';
                        document.getElementById('latex-code').textContent = latex;

                        // Render with MathJax
                        const rendered = document.getElementById('latex-rendered');
                        rendered.innerHTML = '\\\\(' + latex + '\\\\)';
                        MathJax.typesetPromise([rendered]).catch(err => {
                            rendered.textContent = '(Could not render: ' + latex + ')';
                        });
                    }
                })
                .catch(err => {
                    document.getElementById('loading').style.display = 'none';
                    alert('Request failed: ' + err);
                });
        }

        function copyLatex() {
            const text = document.getElementById('latex-code').textContent;
            navigator.clipboard.writeText(text).then(() => {
                const btn = document.getElementById('copy-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy', 2000);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert("L")

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        mask = torch.zeros(
            1, img_tensor.shape[2], img_tensor.shape[3],
            dtype=torch.bool
        ).to(DEVICE)

        print(f"Image size: {image.size}, tensor: {img_tensor.shape}")
        print("Running beam search...")

        with torch.no_grad():
            hyps = model.tamer_model.beam_search(
                img_tensor, mask, **model.hparams
            )

        latex = vocab.indices2label(hyps[0].seq)
        print(f"Result: {latex}")
        return jsonify({'latex': latex})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, port=5000)