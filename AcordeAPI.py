import librosa
import numpy as np
import scipy.ndimage
import os
import yt_dlp
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Chord Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicaRequest(BaseModel):
    url: str

# ==========================================
# 2. GABARITOS DE ACORDES DEFENSIVOS (Com multas matemáticas)
# ==========================================
notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Se uma nota que "destrói" o acorde estiver presente, ela aplica uma multa negativa.
# Ex: Um acorde Maior não pode ter terça menor (-0.8).
template_maior_base = np.array([1.2, 0, 0, -0.8, 1.0, 0, 0, 1.0, 0, 0, 0, 0]) 
template_menor_base = np.array([1.2, 0, 0, 1.0, -0.8, 0, 0, 1.0, 0, 0, 0, 0]) 

# Sétimas e Sextas (Multando as notas conflitantes)
template_7_base     = np.array([1.0, 0, 0, -0.8, 0.9, 0, 0, 0.8, 0, 0, 0.9, -0.8]) 
template_maj7_base  = np.array([1.0, 0, 0, -0.8, 0.9, 0, 0, 0.8, 0, 0, -0.8, 0.9]) 
template_m7_base    = np.array([1.0, 0, 0, 0.9, -0.8, 0, 0, 0.8, 0, 0, 0.9, -0.8]) 
template_6_base     = np.array([1.0, 0, 0, -0.8, 0.9, 0, 0, 0.8, 0, 0.9, -0.8, -0.8]) 

# Acidentes (Os maiores causadores de falsos positivos na voz/violão)
# Sus4: PROIBIDO ter terça maior ou menor
template_sus4_base  = np.array([1.2, 0, 0, -1.0, -1.0, 1.0, 0, 1.0, 0, 0, 0, 0]) 
# Diminuto: PROIBIDO ter quinta justa (Isso impede que acordes menores virem diminutos)
template_dim_base   = np.array([1.2, 0, 0, 1.0, -0.5, 0, 1.0, -1.0, 0, 0, 0, 0]) 
# Aumentado: PROIBIDO ter quinta justa
template_aug_base   = np.array([1.2, 0, 0, -0.5, 1.0, 0, 0, -1.0, 1.0, 0, 0, 0]) 

templates_acordes, nomes_acordes = [], []
for i in range(12):
    templates_acordes.append(np.roll(template_maior_base, i)); nomes_acordes.append(notas[i])
    templates_acordes.append(np.roll(template_menor_base, i)); nomes_acordes.append(f"{notas[i]}m")
    templates_acordes.append(np.roll(template_7_base, i));     nomes_acordes.append(f"{notas[i]}7")
    templates_acordes.append(np.roll(template_maj7_base, i));  nomes_acordes.append(f"{notas[i]}maj7")
    templates_acordes.append(np.roll(template_m7_base, i));    nomes_acordes.append(f"{notas[i]}m7")
    
    templates_acordes.append(np.roll(template_dim_base, i));   nomes_acordes.append(f"{notas[i]}dim")
    templates_acordes.append(np.roll(template_aug_base, i));   nomes_acordes.append(f"{notas[i]}aug")
    templates_acordes.append(np.roll(template_sus4_base, i));  nomes_acordes.append(f"{notas[i]}sus4")
    templates_acordes.append(np.roll(template_6_base, i));     nomes_acordes.append(f"{notas[i]}6")

matriz_acordes = np.array(templates_acordes)

# Gabaritos de Tonalidade continuam os mesmos...


# Gabaritos de Tonalidade (Krumhansl-Schmuckler)
perfil_tom_maior = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
perfil_tom_menor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

templates_tons, nomes_tons = [], []
for i in range(12):
    templates_tons.append(np.roll(perfil_tom_maior, i))
    nomes_tons.append(notas[i])
for i in range(12):
    templates_tons.append(np.roll(perfil_tom_menor, i))
    nomes_tons.append(f"{notas[i]}m")

matriz_tons = np.array(templates_tons)

def identificar_acorde(vetor_chroma):
    pontuacoes = np.dot(matriz_acordes, vetor_chroma)
    return nomes_acordes[np.argmax(pontuacoes)]

# ==========================================
# 3. LÓGICA DE ÁUDIO
# ==========================================
def baixar_audio(url_youtube, nome_arquivo):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{nome_arquivo}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio', 
            'preferredcodec': 'wav', 
            'preferredquality': '192'
        }],
        'quiet': True, 
        'no_warnings': True,
        # Parâmetros adicionados para burlar o bloqueio 403 do YouTube simulando um navegador real
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
        'extractor-args': {'youtube': {'player-client': ['android', 'web']}},
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_youtube])
        
    return f"{nome_arquivo}.wav"

def analisar_musica(caminho_arquivo):
    # Carrega 30 segundos de áudio para análise
    y, sr = librosa.load(caminho_arquivo, duration=30.0)
    
    # ==========================================
    # 1. CÁLCULO DE AFINAÇÃO E BPM (Intacto)
    # ==========================================
    desvio_afinacao = librosa.estimate_tuning(y=y, sr=sr)
    desvio_final = float(desvio_afinacao[0]) if isinstance(desvio_afinacao, np.ndarray) else float(desvio_afinacao)
    frequencia_a4 = 440.0 * (2.0 ** (desvio_final / 12.0))
    
    tempo_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm_final = float(tempo_bpm[0]) if isinstance(tempo_bpm, np.ndarray) else float(tempo_bpm)
    
    # ==========================================
    # 2. CROMAGRAMAS E FUSÃO (Focado e Seguro)
    # ==========================================
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma_geral = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=desvio_final)
    
    # Filtro em E2 para ignorar ruídos graves da madeira do violão
    chroma_graves = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr, 
        tuning=desvio_final,
        fmin=librosa.note_to_hz('E2'), 
        n_octaves=2
    )
    
    # Trava de segurança para garantir o mesmo tamanho das matrizes
    tamanho_minimo = min(chroma_geral.shape[1], chroma_graves.shape[1])
    chroma_geral_cortado = chroma_geral[:, :tamanho_minimo]
    chroma_graves_cortado = chroma_graves[:, :tamanho_minimo]
    
    # Fusão: As notas graves ganham peso 2
    chromagrama_fundido = chroma_geral_cortado + (chroma_graves_cortado * 2.0)
    
    # ==========================================
    # 3. TONALIDADE
    # ==========================================
    soma_chroma = np.sum(chroma_geral_cortado, axis=1)
    pontuacoes_tons = np.dot(matriz_tons, soma_chroma)
    tonalidade = nomes_tons[np.argmax(pontuacoes_tons)]

    # ==========================================
    # 4. EXTRAÇÃO DE ACORDES
    # ==========================================
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

    tempo_final = tempos[-1]
    if acorde_anterior is not None:
        cifra_resultado.append({
            "acorde": acorde_anterior,
            "inicio": round(tempo_inicio, 1),
            "fim": round(tempo_final, 1)
        })
        
    return {
        "bpm": round(bpm_final),
        "tonalidade": tonalidade,
        "afinacao_hz": round(frequencia_a4, 1),
        "cifra": cifra_resultado
    }
# ==========================================
# 4. ROTA DA API
# ==========================================
@app.post("/extrair_acordes")
def extrair_acordes_endpoint(requisicao: MusicaRequest):
    nome_temporario = f"audio_{uuid.uuid4().hex}"
    caminho_arquivo = ""
    try:
        caminho_arquivo = baixar_audio(requisicao.url, nome_temporario)
        resultado = analisar_musica(caminho_arquivo)
        return {"status": "sucesso", "dados": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if caminho_arquivo and os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)