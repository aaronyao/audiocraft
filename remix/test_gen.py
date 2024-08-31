import os
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def test_generate():
    descriptions = ['disco progressive funk with groovy rhythm and spontaneous ad-libs']
    MODEL.set_generation_params(duration=45) 
    generated_outputs = MODEL.generate(descriptions, progress=True)

    save_outputs(generated_outputs)

def test_generate_continuation():
    # 读取 WAV 文件
    sample_wav_path = 'get-lucky-cuts0-14.837551020408164.wav'
    sample_wav, sr = torchaudio.load(sample_wav_path)

    # 如果采样率不匹配，需要进行重采样
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        sample_wav = resampler(sample_wav)

    descriptions = ['disco progressive funk with groovy rhythm and spontaneous ad-libs']
    MODEL.set_generation_params(duration=60) 
    generated_outputs = MODEL.generate_continuation(prompt=sample_wav, 
                                                prompt_sample_rate=SAMPLE_RATE,
                                                descriptions=descriptions,
                                                progress=True)

    save_outputs(generated_outputs)

def test_generate_with_chroma():
    # 读取 WAV 文件
    melody_wav_path = 'my_humming.wav'
    melody_wav, sr = torchaudio.load(melody_wav_path)

    # 如果采样率不匹配，需要进行重采样
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        melody_wav = resampler(melody_wav)

    descriptions = ['piano', 'drums', 'bass']
    MODEL.set_generation_params(duration=30) 
    generated_outputs = MODEL.generate_with_chroma(descriptions=descriptions,
                                            melody_wavs=[melody_wav]*len(descriptions),
                                            melody_sample_rate=SAMPLE_RATE,
                                            progress=True)
    save_outputs(generated_outputs)
    
def test_generate_continuation_with_chroma():
    sample_wav_path = 'sample_drums.wav'
    sample_wav, sample_sr = torchaudio.load(sample_wav_path)
    
    melody_wav_path = 'melody_guitar.wav'
    melody_wav, melody_sr = torchaudio.load(melody_wav_path)

    # 如果采样率不匹配，需要进行重采样
    if sample_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_sr, new_freq=SAMPLE_RATE)
        sample_wav = resampler(sample_wav)
    if melody_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=melody_sr, new_freq=SAMPLE_RATE)
        melody_wav = resampler(melody_wav)

    descriptions = ['funk disco']
    
    melodies = [melody_wav]
    MODEL.set_generation_params(duration=180) 

    generated_outputs = MODEL.generate_continuation_with_chroma(
                            descriptions=descriptions,
                            prompt_wav=sample_wav, prompt_sample_rate=SAMPLE_RATE,
                            melody_wavs=melodies, melody_sample_rate=SAMPLE_RATE, progress=True,return_tokens=False)
    save_outputs(generated_outputs)

def save_outputs(generated_outputs, output_dir="outputs", sample_rate=32000):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个生成的输出并保存为 WAV 文件
    for i, output in enumerate(generated_outputs):
        output_path = os.path.join(output_dir, f"output_{i + 1}.wav")
        audio_write(
            output_path, output, sample_rate, strategy="loudness",
            loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
        
        print(f'wav saved: {output_path}')

import torchaudio
import librosa
import matplotlib.pyplot as plt

def plot_chroma():
    y, sr = librosa.load('tmps9hcgvf7.wav')
    # 计算 Chroma 图谱
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 绘制 Chroma 图谱
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Chroma Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Chroma')
    plt.show()

if __name__ == '__main__':
    SAMPLE_RATE = 32000
    # MODEL = MusicGen.get_pretrained('facebook/musicgen-small')
    # MODEL = MusicGen.get_pretrained('facebook/musicgen-large')
    # test_generate()
    # test_generate_continuation()

    MODEL = MusicGen.get_pretrained('facebook/musicgen-melody')
    test_generate_with_chroma()

    # plot_chroma()
