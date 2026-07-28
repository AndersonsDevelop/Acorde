import os
import uuid
import uvicorn
import librosa
import numpy as np
import scipy.ndimage
import ffmpeg_downloader as ffdl
import traceback
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    ffdl.add_path()
except Exception:
    ffdl.download_ffmpeg()
    ffdl.add_path()

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Chord Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

templates_acordes = []
nomes_acordes = []

def criar_template(base, i):
    arr = np.roll(base, i)
    return np.array(arr, dtype=float)

for i in range(12):
    templates_acordes.append(criar_template([1.2, 0, 0, -0.8, 1.0, 0, 0, 1.0, 0, 0, 0, 0], i))
    nomes_acordes.append(notas[i])
    templates_acordes.append(criar_template([1.2, 0, 0, 1.0, -0.8, 0, 0, 1.0, 0, 0, 0, 0], i))
    nomes_acordes.append(f"{notas[i]}m")
    templates_acordes.append(criar_template([1.0, 0, 0, -0.8, 0.9, 0, 0, 0.8, 0, 0, 0.9, -0.8], i))
    nomes_acordes.append(f"{notas[i]}7")
    templates_acordes.append(criar_template([1.0, 0, 0, -0.8, 0.9, 0, 0, 0.8, 0, 0, -0.8, 0.9], i))
    nomes_acordes.append(f"{notas[i]}maj7")
    templates_acordes.append(criar_template([1.0, 0, 0, 0.9, -0.8, 0, 0, 0.8, 0, 0, 0.9, -0.8], i))
    nomes_acordes.append(f"{notas[i]}m7")
    templates_acordes.append(criar_template([1.2, 0, 0, 1.0, -0.5, 0, 1.0, -1.0, 0, 0, 0, 0], i))
    nomes_acordes.append(f"{notas[i]}dim")
    templates_acordes.append(criar_template([1.2, 0, 0, -0.5, 1.0, 0, 0, -1.0, 1.0, 0, 0, 0], i))
    nomes_acordes.append(f"{notas[i]}aug")
    templates_acordes.append(criar_template([1.2, 0, 0, -1.0, -1.0, 1.0, 0, 1.0, 0, 0, 0, 0], i))
    nomes_acordes.append(f"{notas[i]}sus4")
    templates_acordes.append(criar_template([1.0, 0, 0, -0.8, 0.9, 0, 0, 0.8, 0, 0.9, -0.8, -0.8], i))
    nomes_acordes.append(f"{notas[i]}6")

matriz_acordes = np.vstack(templates_acordes)

perfil_tom_maior = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
perfil_tom_menor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

templates_tons, nomes_tons = [], []
for i in range(12):
    templates_tons.append(np.roll(perfil_tom_maior, i))
    nomes_tons.append(notas[i])
for i in range(12):
    templates_tons.append(np.roll(perfil_tom_menor, i))
    nomes_tons.append(f"{notas[i]}m")

matriz_tons = np.vstack(templates_tons)

def identificar_acorde(vetor_chroma):
    pontuacoes = np.dot(matriz_acordes, vetor_chroma)
    return nomes_acordes[np.argmax(pontuacoes)]

def analisar_musica(caminho_arquivo):
    y, sr = librosa.load(caminho_arquivo, sr=11025, mono=True, duration=30.0)
    
    desvio_afinacao = librosa.estimate_tuning(y=y, sr=sr)
    desvio_final = float(desvio_afinacao[0]) if isinstance(desvio_afinacao, np.ndarray) else float(desvio_afinacao)
    frequencia_a4 = 440.0 * (2.0 ** (desvio_final / 12.0))
    
    tempo_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm_final = float(tempo_bpm[0]) if isinstance(tempo_bpm, np.ndarray) else float(tempo_bpm)
    
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma_geral = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=desvio_final)
    chroma_graves = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=desvio_final, fmin=librosa.note_to_hz('E2'), n_octaves=2)
    
    tamanho_minimo = min(chroma_geral.shape[1], chroma_graves.shape[1])
    chroma_geral_cortado = chroma_geral[:, :tamanho_minimo]
    chroma_graves_cortado = chroma_graves[:, :tamanho_minimo]
    
    chromagrama_fundido = chroma_geral_cortado + (chroma_graves_cortado * 2.0)
    
    soma_chroma = np.sum(chroma_geral_cortado, axis=1)
    pontuacoes_tons = np.dot(matriz_tons, soma_chroma)
    tonalidade = nomes_tons[np.argmax(pontuacoes_tons)]

    chromagrama_suave = scipy.ndimage.median_filter(chromagrama_fundido, size=(1, 15))
    tempos = librosa.frames_to_time(np.arange(chromagrama_suave.shape[1]), sr=sr)
    
    salto_frames = 43 
    acorde_anterior = None
    tempo_inicio = 0.0
    cifra_resultado = []
    
    for i in range(0, chromagrama_suave.shape[1], salto_frames):
        frame_atual = chromagrama_suave[:, i]
        acorde_detectado = identificar_acorde(frame_atual)
        tempo_segundos = tempos[i]
        
        if acorde_detectado != acorde_anterior:
            if acorde_anterior is not None:
                cifra_resultado.append({
                    "acorde": acorde_anterior,
                    "inicio": round(tempo_inicio, 1),
                    "fim": round(tempo_segundos, 1)
                })
            acorde_anterior = acorde_detectado
            tempo_inicio = tempo_segundos

    if acorde_anterior is not None:
        cifra_resultado.append({
            "acorde": acorde_anterior,
            "inicio": round(tempo_inicio, 1),
            "fim": round(tempos[-1], 1)
        })
        
    return {
        "bpm": round(bpm_final),
        "tonalidade": tonalidade,
        "afinacao_hz": round(frequencia_a4, 1),
        "cifra": cifra_resultado
    }

@app.post("/extrair_acordes")
async def extrair_acordes_endpoint(file: UploadFile = File(...)):
    nome_temporario = f"audio_{uuid.uuid4().hex}_{file.filename}"
    try:
        contents = await file.read()
        with open(nome_temporario, "wb") as f:
            f.write(contents)
            
        resultado = analisar_musica(nome_temporario)
        resultado["titulo"] = file.filename
        return {"status": "sucesso", "dados": resultado}
    except Exception as e:
        print("ERRO DETECTADO NO BACKEND:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(nome_temporario):
            os.remove(nome_temporario)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)