"""Standalone Gradio app to load a saved PyTorch checkpoint (model.ckpt or model_state_dict.pt)
and scalers.pt produced by the notebook, then serve predictions.

Usage:
    python gradio_app.py

Ensure you have the trained artifacts in the same folder:
  - model.ckpt  (Architecture.save_checkpoint format)
  - or model_state_dict.pt
  - scalers.pt  (a dict with keys 'mu','std','y_mu','y_std' saved with torch.save)

If `feature_cols.json` exists it will be used for input labels. Otherwise the script will
attempt to infer feature names from `listings.csv` or fall back to numbered inputs.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

try:
    import gradio as gr
except Exception:
    raise SystemExit("É necessário instalar o Gradio para rodar este app. Instale com: pip install gradio")

ROOT = os.path.dirname(__file__)

def load_scalers(path=os.path.join(ROOT, 'scalers.pt')):
    if not os.path.exists(path):
        print('Arquivo scalers.pt não encontrado em', path)
        return None
    # torch 2.6+ may restrict unpickling by default. Try normal load first and
    # if it fails, retry with a safe_globals context to allow numpy reconstruction.
    try:
        scalers = torch.load(path, map_location='cpu')
    except Exception as e:
        print('torch.load falhou ao carregar scalers.pt com erro:', e)
        try:
            # Try to allowlist numpy reconstruction functions (trusted file required)
            import numpy as _np
            try:
                # safe_globals context manager is available in torch.serialization
                with torch.serialization.safe_globals([_np.core.multiarray._reconstruct]):
                    scalers = torch.load(path, map_location='cpu', weights_only=False)
            except Exception:
                # Older/newer torch versions may expose add_safe_globals instead
                try:
                    torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct])
                    scalers = torch.load(path, map_location='cpu', weights_only=False)
                finally:
                    # best-effort: proceed even if removal isn't available
                    pass
        except Exception as e2:
            print('Tentativa alternativa para carregar scalers falhou:', e2)
            return None
    # convert tensors to numpy if needed
    for k, v in list(scalers.items()):
        if isinstance(v, torch.Tensor):
            scalers[k] = v.detach().cpu().numpy()
    return scalers

def infer_feature_cols():
    # Prefer explicit JSON
    fn_json = os.path.join(ROOT, 'feature_cols.json')
    if os.path.exists(fn_json):
        try:
            with open(fn_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    # Fall back to listings.csv (best-effort)
    fn_csv = os.path.join(ROOT, 'listings.csv')
    if os.path.exists(fn_csv):
        try:
            df = pd.read_csv(fn_csv)
            cols = [c for c in df.columns if c != 'price']
            print('Colunas de features inferidas a partir de listings.csv (melhor tentativa).')
            return cols
        except Exception:
            pass

    # Generic fallback: 10 numeric inputs
    print('Não foi possível inferir nomes das features; usando entradas numéricas genéricas (10).')
    return [f'x{i}' for i in range(10)]

def build_model(input_dim):
    # O notebook usou um modelo linear simples: Sequential(Linear(D,1)).
    return nn.Sequential(nn.Linear(input_dim, 1))

def load_model(root=ROOT):
    # Try model.ckpt (Architecture.save_checkpoint), then model_state_dict.pt
    ckpt_path = os.path.join(root, 'model.ckpt')
    state_path = os.path.join(root, 'model_state_dict.pt')
    scalers = load_scalers()
    feature_cols = infer_feature_cols()
    D = len(feature_cols)

    model = build_model(D)
    loaded = False
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location='cpu')
            if 'model_state_dict' in ck:
                model.load_state_dict(ck['model_state_dict'])
                loaded = True
                print('Modelo carregado de', ckpt_path)
        except Exception as e:
            print('Falha ao carregar', ckpt_path, '->', e)

    if not loaded and os.path.exists(state_path):
        try:
            sd = torch.load(state_path, map_location='cpu')
            model.load_state_dict(sd)
            loaded = True
            print('model_state_dict carregado de', state_path)
        except Exception as e:
            print('Falha ao carregar', state_path, '->', e)

    if not loaded:
        print('Aviso: nenhum peso do modelo foi carregado. Usando modelo com pesos aleatórios.')

    model.eval()
    return model, scalers, feature_cols

def predict_fn(model, scalers, feature_cols, *vals):
    x = np.array(vals, dtype=np.float32).reshape(1, -1)
    # Aplica normalização de features se disponível (espera scalers['mu'], scalers['std'] como arrays 1D)
    if scalers is not None and 'mu' in scalers and 'std' in scalers:
        mu = np.asarray(scalers['mu']).reshape(1, -1)
        std = np.asarray(scalers['std']).reshape(1, -1)
        try:
            x = (x - mu) / (std + 1e-12)
        except Exception:
            # fallback: try broadcasting
            x = (x - mu) / (std + 1e-12)

    xt = torch.from_numpy(x).float()
    with torch.no_grad():
        y = model(xt).cpu().numpy().squeeze()

    # desfaz transformação do target se disponível
    if scalers is not None and 'y_mu' in scalers and 'y_std' in scalers:
        y = y * float(scalers['y_std']) + float(scalers['y_mu'])

    return float(y)

def make_interface():
    model, scalers, feature_cols = load_model()
    inputs = [gr.Number(label=c, value=0.0) for c in feature_cols]
    iface = gr.Interface(
        fn=lambda *vals: predict_fn(model, scalers, feature_cols, *vals),
        inputs=inputs,
        outputs=gr.Number(label='Preço previsto (R$)'),
        title='Previsão de preços - Rio de Janeiro',
        description=(
            'Modelo de regressão treinado no dataset do Rio de Janeiro. '
            'Carrega artefatos de model.ckpt/model_state_dict.pt e scalers.pt'
        ),
    )
    return iface

def main():
    iface = make_interface()
    print('Iniciando app Gradio em http://localhost:7860 (Previsão de preços — Rio de Janeiro)')
    iface.launch(server_name='0.0.0.0', server_port=7860, share=False)

if __name__ == '__main__':
    main()
