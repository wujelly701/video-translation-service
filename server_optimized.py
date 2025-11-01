import os
import io
import time
import sys
import threading
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from typing import Optional

# 强制输出无缓冲
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

# Lazy imports to speed cold start
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    M2M100ForConditionalGeneration = None
    M2M100Tokenizer = None
    torch = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_LOCK = threading.Lock()
ASR_MODEL = None
TRANSLATION_MODEL = None
TOKENIZER = None

# Readiness booleans kept for backward compatibility
ASR_READY = False
TRANSLATION_READY = False

# Extended status object for progress reporting
STATUS = {
    'phase': 'idle',               # idle | asr_downloading | asr_loading | asr_ready | translation_downloading | translation_loading | translation_ready | ready | error
    'progress': 0.0,               # 0.0 ~ 1.0 coarse grained
    'message': 'Idle',
    'started_at': None,
    'error': None,
}

LOADING_THREAD: Optional[threading.Thread] = None
LOADING_PARAMS = {}

def _set_status(phase: str, progress: float, message: str):
    STATUS['phase'] = phase
    STATUS['progress'] = round(max(0.0, min(1.0, progress)), 3)
    STATUS['message'] = message
    STATUS['error'] = None if phase != 'error' else STATUS.get('error')

def _set_error(err_msg: str):
    STATUS['phase'] = 'error'
    STATUS['message'] = err_msg
    STATUS['error'] = err_msg

def _load_models_async(asr_size: str, translation_name: str, device: str, compute_type: str):
    """Background thread target to load models with phase updates."""
    global ASR_MODEL, TRANSLATION_MODEL, TOKENIZER, ASR_READY, TRANSLATION_READY, LOADING_THREAD
    try:
        if STATUS['started_at'] is None:
            STATUS['started_at'] = time.time()
        # ASR model
        if WhisperModel and not ASR_READY:
            _set_status('asr_downloading', 0.02, f'Downloading/Preparing ASR model {asr_size} ...')
            try:
                # faster-whisper downloads inside constructor
                _set_status('asr_loading', 0.08, f'Loading ASR model {asr_size} ...')
                ASR_MODEL = WhisperModel(asr_size, device=device, compute_type=compute_type, download_root='./models/whisper')
                ASR_READY = True
                _set_status('asr_ready', 0.35, f'ASR model ready ({asr_size}).')
            except Exception as e:
                _set_error(f'ASR load failed: {e}')
                return
        elif ASR_READY:
            _set_status('asr_ready', max(STATUS['progress'], 0.35), 'ASR already ready.')
        # Translation model - 使用m2m100或NLLB
        if not TRANSLATION_READY:
            # 检测是否是m2m100模型
            is_m2m100 = 'm2m100' in translation_name.lower()
            _set_status('translation_downloading', 0.4, f'Downloading translation model {translation_name} ...')
            try:
                if is_m2m100 and M2M100Tokenizer and M2M100ForConditionalGeneration:
                    print(f'[Translation] Using M2M100 model: {translation_name}')
                    TOKENIZER = M2M100Tokenizer.from_pretrained(translation_name, cache_dir='./models/m2m100')
                    _set_status('translation_loading', 0.55, f'Loading translation model {translation_name} ...')
                    TRANSLATION_MODEL = M2M100ForConditionalGeneration.from_pretrained(translation_name, cache_dir='./models/m2m100')
                elif AutoTokenizer and AutoModelForSeq2SeqLM:
                    print(f'[Translation] Using NLLB model: {translation_name}')
                    TOKENIZER = AutoTokenizer.from_pretrained(translation_name, cache_dir='./models/nllb')
                    _set_status('translation_loading', 0.55, f'Loading translation model {translation_name} ...')
                    TRANSLATION_MODEL = AutoModelForSeq2SeqLM.from_pretrained(translation_name, cache_dir='./models/nllb')
                else:
                    raise ImportError('No translation library available')
                    
                if torch and device == 'cuda':
                    TRANSLATION_MODEL = TRANSLATION_MODEL.to(device)
                TRANSLATION_READY = True
                _set_status('translation_ready', 0.85, f'Translation model ready ({translation_name}).')
                print(f'[Translation] Model loaded successfully: {translation_name}')
            except Exception as e:
                _set_error(f'Translation load failed: {e}')
                print(f'[Translation] Error loading model: {e}')
                import traceback
                traceback.print_exc()
                return
        elif TRANSLATION_READY:
            _set_status('translation_ready', max(STATUS['progress'], 0.85), 'Translation already ready.')
        if ASR_READY and TRANSLATION_READY:
            _set_status('ready', 1.0, 'All models ready.')
    finally:
        LOADING_THREAD = None

# Simple health endpoint
@app.route('/health', methods=['GET'])
def health():
    # Backward compatible booleans plus extended status
    return jsonify({
        'status': 'ok',
        'asr_ready': ASR_READY,
        'translation_ready': TRANSLATION_READY,
        'phase': STATUS['phase'],
        'progress': STATUS['progress'],
        'message': STATUS['message'],
        'error': STATUS['error'],
        'ready': ASR_READY and TRANSLATION_READY
    })

@app.route('/init', methods=['POST'])
def init_models():
    global LOADING_THREAD, LOADING_PARAMS
    payload = request.json or {}
    asr_size = payload.get('asr_model_size', 'medium')
    translation_name = payload.get('translation_model', 'facebook/nllb-200-distilled-600M')
    device = 'cuda' if payload.get('use_gpu', True) else 'cpu'
    compute_type = payload.get('compute_type', 'float16' if device == 'cuda' else 'int8')

    # If already ready return immediately
    if ASR_READY and TRANSLATION_READY:
        return jsonify({'success': True, 'already_ready': True, 'asr_ready': ASR_READY, 'translation_ready': TRANSLATION_READY, 'phase': STATUS['phase'], 'progress': STATUS['progress']})

    # Avoid spawning multiple threads
    with MODEL_LOCK:
        if LOADING_THREAD is None:
            _set_status('starting', 0.0, 'Starting model load thread ...')
            LOADING_PARAMS = {
                'asr_size': asr_size,
                'translation_name': translation_name,
                'device': device,
                'compute_type': compute_type,
            }
            LOADING_THREAD = threading.Thread(target=_load_models_async, args=(asr_size, translation_name, device, compute_type), daemon=True)
            LOADING_THREAD.start()
        else:
            # Thread already running; optionally update desired params if changed
            LOADING_PARAMS.update({'asr_size': asr_size, 'translation_name': translation_name})
    return jsonify({'success': True, 'started': True, 'phase': STATUS['phase'], 'progress': STATUS['progress']})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not ASR_READY:
        return jsonify({'success': False, 'error': 'ASR not ready'}), 400
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file'}), 400
    f = request.files['audio']
    # ✅ 生成唯一文件名避免并发冲突
    import uuid
    ext = os.path.splitext(secure_filename(f.filename or 'audio.bin'))[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    
    language = request.form.get('language')  # optional
    started = time.time()
    try:
        # 如果是PCM，转换为WAV
        if filename.endswith('.pcm') or filename.endswith('.raw'):
            import wave
            wav_path = path.replace('.pcm', '.wav').replace('.raw', '.wav')
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                with open(path, 'rb') as pcm_file:
                    wav_file.writeframes(pcm_file.read())
            os.remove(path)
            path = wav_path
        
        # ✅ 启用VAD，在语音边界截断，避免在单词中间切断
        segments, info = ASR_MODEL.transcribe(
            path, 
            language=language, 
            vad_filter=True,  # 使用VAD检测静音点，在语音边界截断
            beam_size=5,
            best_of=5
        )
        
        text = ''
        seg_list = []
        for s in segments:
            text += s.text
            seg_list.append({'start': s.start, 'end': s.end, 'text': s.text})
        os.remove(path)
        return jsonify({'success': True, 'text': text.strip(), 'language': info.language, 'language_probability': info.language_probability, 'segments': seg_list, 'processing_time_ms': int((time.time()-started)*1000)})
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    if not TRANSLATION_READY:
        return jsonify({'success': False, 'error': 'Translation not ready'}), 400
    data = request.json or {}
    text = data.get('text')
    src_lang = data.get('source_language')
    tgt_lang = data.get('target_language')
    if not text:
        return jsonify({'success': False, 'error': 'No text'}), 400
    if not src_lang or not tgt_lang:
        return jsonify({'success': False, 'error': 'Missing languages'}), 400
    
    # 语言代码映射 - 支持NLLB和M2M100
    # M2M100使用简单的ISO代码（zh, en等）
    # NLLB使用扩展代码（zho_Hans, eng_Latn等）
    nllb_lang_map = {
        'zh': 'zho_Hans', 'en': 'eng_Latn', 'ja': 'jpn_Jpan', 
        'ko': 'kor_Hang', 'de': 'deu_Latn', 'fr': 'fra_Latn',
        'es': 'spa_Latn', 'ru': 'rus_Cyrl', 'ar': 'arb_Arab'
    }
    m2m100_lang_map = {
        'zh': 'zh', 'en': 'en', 'ja': 'ja', 'ko': 'ko', 
        'de': 'de', 'fr': 'fr', 'es': 'es', 'ru': 'ru', 'ar': 'ar'
    }
    
    # 检测模型类型
    is_m2m100 = isinstance(TOKENIZER, M2M100Tokenizer) if M2M100Tokenizer else False
    
    try:
        # 使用锁防止并发访问模型（修复"Already borrowed"错误）
        with MODEL_LOCK:
            if is_m2m100:
                # M2M100翻译流程
                src_code = m2m100_lang_map.get(src_lang, src_lang)
                tgt_code = m2m100_lang_map.get(tgt_lang, tgt_lang)
                TOKENIZER.src_lang = src_code
                inputs = TOKENIZER(text, return_tensors='pt')
                if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                # M2M100使用forced_bos_token_id指定目标语言
                tgt_lang_id = TOKENIZER.get_lang_id(tgt_code)
                gen_tokens = TRANSLATION_MODEL.generate(
                    **inputs, 
                    forced_bos_token_id=tgt_lang_id,
                    max_length=512, 
                    num_beams=3,
                    early_stopping=True
                )
                decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                # NLLB翻译流程
                src_code = nllb_lang_map.get(src_lang, src_lang)
                tgt_code = nllb_lang_map.get(tgt_lang, tgt_lang)
                TOKENIZER.src_lang = src_code
                inputs = TOKENIZER(text, return_tensors='pt')
                if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                gen_tokens = TRANSLATION_MODEL.generate(
                    **inputs, 
                    forced_bos_token_id=TOKENIZER.convert_tokens_to_ids(tgt_code), 
                    max_length=512, 
                    num_beams=3
                )
                decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
        
        if not decoded:
            return jsonify({'success': False, 'error': 'Translation produced empty result'}), 500
        translated = decoded[0]
        return jsonify({'success': True, 'translated_text': translated})
    except Exception as e:
        import traceback
        error_msg = f"Translation error: {str(e)}"
        print(f"[Translation Error] {error_msg}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/process', methods=['POST'])
def process():
    # full pipeline: transcribe + translate
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file'}), 400
    tgt_lang = request.form.get('target_language', 'en')
    src_lang_override = request.form.get('source_language')
    # ✅ 如果source_language是'auto'，不使用override
    if src_lang_override == 'auto':
        src_lang_override = None
    # Step ASR
    trans_resp = transcribe()
    if isinstance(trans_resp, tuple):
        return trans_resp  # error
    data = trans_resp.get_json()
    if not data.get('success'):
        return trans_resp
    source_lang = src_lang_override or data.get('language')
    if source_lang == tgt_lang:
        return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': data.get('text'), 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments')})
    # Translation
    if TRANSLATION_READY:
        # 构造翻译请求数据
        translate_data = {
            'text': data.get('text'),
            'source_language': source_lang,
            'target_language': tgt_lang
        }
        # 调用翻译逻辑（支持M2M100和NLLB）
        try:
            # 语言代码映射
            nllb_lang_map = {
                'zh': 'zho_Hans', 'en': 'eng_Latn', 'ja': 'jpn_Jpan', 
                'ko': 'kor_Hang', 'de': 'deu_Latn', 'fr': 'fra_Latn',
                'es': 'spa_Latn', 'ru': 'rus_Cyrl', 'ar': 'arb_Arab'
            }
            m2m100_lang_map = {
                'zh': 'zh', 'en': 'en', 'ja': 'ja', 'ko': 'ko', 
                'de': 'de', 'fr': 'fr', 'es': 'es', 'ru': 'ru', 'ar': 'ar'
            }
            
            # 检测模型类型
            is_m2m100 = isinstance(TOKENIZER, M2M100Tokenizer) if M2M100Tokenizer else False
            
            if is_m2m100:
                # M2M100翻译流程
                src_code = m2m100_lang_map.get(source_lang, source_lang)
                tgt_code = m2m100_lang_map.get(tgt_lang, tgt_lang)
                TOKENIZER.src_lang = src_code
                inputs = TOKENIZER(translate_data['text'], return_tensors='pt')
                if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                tgt_lang_id = TOKENIZER.get_lang_id(tgt_code)
                gen_tokens = TRANSLATION_MODEL.generate(
                    **inputs, 
                    forced_bos_token_id=tgt_lang_id,
                    max_length=512, 
                    num_beams=3,
                    early_stopping=True
                )
                decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                # NLLB翻译流程
                src_code = nllb_lang_map.get(source_lang, source_lang)
                tgt_code = nllb_lang_map.get(tgt_lang, tgt_lang)
                TOKENIZER.src_lang = src_code
                inputs = TOKENIZER(translate_data['text'], return_tensors='pt')
                if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                gen_tokens = TRANSLATION_MODEL.generate(
                    **inputs, 
                    forced_bos_token_id=TOKENIZER.convert_tokens_to_ids(tgt_code), 
                    max_length=512, 
                    num_beams=3
                )
                decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
            
            if not decoded:
                raise ValueError('Translation produced empty result')
            translated_text = decoded[0]
            
            return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': translated_text, 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments')})
        except Exception as e:
            import traceback
            print(f"[Translation Error in /process] {str(e)}")
            traceback.print_exc()
            # 失败后fallback到源文本
            return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': data.get('text'), 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments'), 'note': 'translation error, fallback to source'})
    # Fallback: return source only
    return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': data.get('text'), 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments'), 'note': 'translation fallback'})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=50515, type=int)
    parser.add_argument('--asr_model_size', default='medium')
    parser.add_argument('--translation_model', default='facebook/nllb-200-distilled-1.3B')  # 升级到1.3B（BLEU +8-10分）
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()
    print('[PythonService] Starting with config:', args)
    print('[PythonService] HuggingFace endpoint:', os.environ.get('HF_ENDPOINT') or 'default')
    sys.stdout.flush()
    
    # 🔥 启动后自动加载模型
    device = 'cpu' if args.no_gpu else 'cuda'
    compute_type = 'int8' if args.no_gpu else 'float16'
    print(f'[PythonService] Auto-loading models: ASR={args.asr_model_size}, Translation={args.translation_model}, Device={device}')
    sys.stdout.flush()
    
    import threading
    def load_models():
        _load_models_async(args.asr_model_size, args.translation_model, device, compute_type)
    
    # 在后台线程加载模型，不阻塞Flask启动
    loader_thread = threading.Thread(target=load_models, daemon=True)
    loader_thread.start()
    
    app.run(host=args.host, port=args.port, debug=False)
