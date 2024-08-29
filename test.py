import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from tempfile import NamedTemporaryFile

def test_generate():
    descriptions = ['disco progressive funk with groovy rhythm and spontaneous ad-libs']
    MODEL.set_generation_params(duration=20) 
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
    melody_wav_path = 'get-lucky-cuts0-14.837551020408164.wav'
    melody_wav, sr = torchaudio.load(melody_wav_path)

    # 如果采样率不匹配，需要进行重采样
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        melody_wav = resampler(melody_wav)

    descriptions = ['disco progressive funk with groovy rhythm and spontaneous ad-libs']
    melodies = [melody_wav]
    MODEL.set_generation_params(duration=30) 
    generated_outputs = MODEL.generate_with_chroma(
                            descriptions=descriptions,melody_wavs=melodies,
                            melody_sample_rate=SAMPLE_RATE, progress=True,return_tokens=False)
    save_outputs(generated_outputs)

def save_outputs(generated_outputs):
    for output in generated_outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, SAMPLE_RATE, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)

            print(f'wav: {file.name}')


if __name__ == '__main__':

    SAMPLE_RATE = 32000
    # MODEL = MusicGen.get_pretrained('facebook/musicgen-small')
    # test_generate_continuation()

    MODEL = MusicGen.get_pretrained('facebook/musicgen-melody')
    test_generate_with_chroma()
