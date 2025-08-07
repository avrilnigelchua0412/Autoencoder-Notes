import os
import librosa
import numpy as np
import h5py

class Loader:
    """Loader is responsible for loading the audio file with error handling and resampling if necessary."""
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        try:
            # Load original with original sample rate
            signal, original_sr = librosa.load(file_path, sr=None, mono=self.mono, duration=self.duration)
            if original_sr != self.sample_rate:
                if original_sr < self.sample_rate:
                    raise ValueError(f"Sample rate too low: {original_sr} < expected {self.sample_rate}")
                # Resample
                signal = librosa.resample(signal, orig_sr=original_sr, target_sr=self.sample_rate)
                    
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")
        
        return signal

class Padder:
    """Padder is responsible to apply padding to an array."""
    # mode='constant' is used for padding because silence (0.0) is natural, neutral, 
    # and non-disruptive in audio — perfect for letting models focus on the real signal.
    def __init__(self, mode="constant"): 
        self.mode = mode
    def right_pad(self, array, num_missing_samples): 
        # Right padding is standard because it preserves the natural flow of time in audio. 
        # The model sees the real signal from the start, and silence is added only after the actual content — just like in real-world recordings.
        padded_array = np.pad(array, (0, num_missing_samples), mode=self.mode)
        return padded_array

class LogSpectrogramExtractor:
    # LogSpectrogramExtractor extracts a log-scaled spectrogram from audio
    def __init__(self, sample_rate, frame_size, hop_length):
        self.sample_rate = sample_rate        # Audio sample rate (e.g., 16000 Hz)
        self.frame_size = frame_size          # Frame size for STFT (e.g., 1024)
        self.hop_length = hop_length          # Step size between frames

    def extract(self, signal):
        stft = librosa.stft(y=signal, n_fft=self.frame_size, hop_length=self.hop_length) # Short-Time Fourier Transform
        # this outputs shape (n_fft/2+1, num_frames) for complex-valued STFT
        # some would just remove the last (n_fft/2+1, num_frames)[:1]
        # but since where generating, completeness > convenience.
        magnitude = np.abs(stft)  # Get magnitude
        log_spectrogram = librosa.amplitude_to_db(magnitude, ref=1.0, top_db=80.0)  # Convert to log scale (dB)
        # Explicitly set the reference to 1.0 and top_db to 80.0
        # ref=1.0 means 0 dB corresponds to an amplitude of 1
        return log_spectrogram

class LogMelSpectrogramExtractor:
    # LogMelSpectrogramExtractor extracts a perceptually meaningful log-Mel spectrogram from audio

    def __init__(self, sample_rate, frame_size, hop_length, n_mels):
        self.sample_rate = sample_rate        # Audio sample rate (e.g., 16000 Hz)
        self.frame_rate = frame_size                    # Frame size for STFT (e.g., 1024)
        self.hop_length = hop_length          # Step size between frames
        self.n_mels = n_mels                  # Number of Mel bands (e.g., 64 or 128)

    def extract(self, signal):
        mel_spectrogram = librosa.feature.melspectrogram(y=signal,
                                                         sr=self.sample_rate,
                                                         n_fft=self.frame_rate,
                                                         hop_length=self.hop_length,
                                                         n_mels=self.n_mels, # Number of Mel bands (e.g., 64, 128). Controls how detailed the Mel filterbank is.
                                                         power=2.0 # Power spectrogram for Mel scaling, By default, power=2 operates on a power spectrum.
                                                         # Calculates power instead of magnitude. (Power = amplitude²)
                                                         )  # Power spectrogram for Mel scaling. Outputs (n_mels, num_frames)
        # Convert to log scale (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0, top_db=80.0) # Explicitly set the reference to 1.0 and top_db to 80.0
        # ref=1.0 means 0 dB corresponds to an amplitude of 1
        return log_mel_spectrogram
    
class MinMaxNormalizer:
    """
    the MinMaxNormalizer normalizes an array to a specified range [min_value, max_value]
    and can denormalize it back to the original range.
    The db or decibel range is specified by db_min and db_max.
    In the librosa functions, the reference value is set to 1.0 and the top_db is set to 80.0.
    If the top_db is set to 80.0 and reference value is set to 1.0 then the db_min should be -80.0 and db_max is 0.0.
    """
    
    def __init__(self, min_value=0.0, max_value=1.0, db_min=-80.0, db_max=0.0):
        self.min_value = min_value
        self.max_value = max_value
        self.db_min = db_min
        self.db_max = db_max
        
    def normalize(self, array):
        return ((array - self.db_min) / (self.db_max - self.db_min)) * (self.max_value - self.min_value) + self.min_value
    
    def denormalize(self, normalized_array):
        return ((normalized_array - self.min_value) / (self.max_value - self.min_value)) * (self.db_max - self.db_min) + self.db_min

class Saver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
            
    def save(self, data, name):
        # Convert to numpy arrays
        data['train'] = np.array(data['train'])
        data['label'] = np.array(data['label'])
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        # Full path to save file
        save_path = os.path.join(self.output_dir, f"{name}_dataset.h5")
        # Save datasets
        with h5py.File(save_path, 'w') as h5f:
            for key, value in data.items():
                h5f.create_dataset(key, data=value)
        
class PreprocessingPipeline:
    """PreprocessingPipeline orchestrates the audio preprocessing steps."""
    def __init__(self):
        self.padder = None
        self.log_spectrogram_extractor = None
        self.log_mel_spectrogram_extractor = None
        self.min_max_normalizer = None
        self.saver = None
        self._loader = None
        self._num_expected_samples = None
        self._normalized_log_spectrogram_data = {
            'train' : [],
            'label' : []
        }
        self._normalized_log_mel_spectrogram_data = {
            'train' : [],
            'label' : []
        }
    
    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        if not isinstance(loader, Loader):
            raise TypeError("Loader must be an instance of Loader class.")
        self._loader = loader
        self._num_expected_samples = int(self.loader.sample_rate * self.loader.duration)
    
    def process(self, file_path):
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith(".wav"):
                    label = file.split('_')[0]
                    file_path = os.path.join(root, file)
                    self.process_file(file_path, int(label))
            print("Processing...")
                    
    def process_file(self, file_path, label):
        try:
            signal = self.loader.load(file_path)
            if self._padding_required(signal):
                # print("Before: ", signal.shape)
                signal = self._apply_padding(signal)
                # print("After: ", signal.shape)
            log_spectrogram = self.log_spectrogram_extractor.extract(signal)
            log_mel_spectrogram = self.log_mel_spectrogram_extractor.extract(signal)
            # print("Log Shape ", log_spectrogram.max())
            # print("Log Mel Shape ", log_mel_spectrogram.max())
            normalized_log_spectrogram = self.min_max_normalizer.normalize(log_spectrogram)
            normalized_log_mel_spectrogram = self.min_max_normalizer.normalize(log_mel_spectrogram)
            # print("Norm Log Shape ", normalized_log_spectrogram.shape)
            # print("Norm Log Mel Shape ", normalized_log_mel_spectrogram.shape)
            self._normalized_log_spectrogram_data['train'].append(normalized_log_spectrogram)
            self._normalized_log_mel_spectrogram_data['train'].append(normalized_log_mel_spectrogram)
            self._normalized_log_spectrogram_data['label'].append(label)
            self._normalized_log_mel_spectrogram_data['label'].append(label)
            # print(type(label), label)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    def _padding_required(self, signal):
        # Check if padding is required based on the signal length and expected duration
        return len(signal) < self._num_expected_samples
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        return self.padder.right_pad(signal, num_missing_samples)
    
    def return_dataset(self):
        return self._normalized_log_spectrogram_data, self._normalized_log_mel_spectrogram_data
    
if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.793875 # In seconds
    SAMPLE_RATE = 22050
    MONO = True
    N_MELS = 128 # 
    OUTPUT_DIR = 'Interpretable Sound Generation/Dataset'
    
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH)
    log_mel_spectrogram_extractor = LogMelSpectrogramExtractor(SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH, N_MELS)
    min_max_normalizer = MinMaxNormalizer()
    preprocessing_pipeline = PreprocessingPipeline()
    saver = Saver(OUTPUT_DIR)
    
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.log_spectrogram_extractor = log_spectrogram_extractor
    preprocessing_pipeline.log_mel_spectrogram_extractor = log_mel_spectrogram_extractor
    preprocessing_pipeline.min_max_normalizer = min_max_normalizer
    
    preprocessing_pipeline.process("Interpretable Sound Generation")
    
    log_spec_data, log_mel_spec_data = preprocessing_pipeline.return_dataset()
    
    saver.save(log_spec_data, 'log_spec_data')
    saver.save(log_mel_spec_data, 'log_mel_spec_data')
    
    # # Determining the "best" duration
    # durations = []
    # for root, dirs, files in os.walk("Interpretable Sound Generation"):
    #     for file in files:
    #         if file.endswith(".wav"):
    #             file_path = os.path.join(root, file)
    #             duration = librosa.get_duration(path=file_path)
    #             durations.append(duration)
                
    # durations = np.array(durations)
    # print("Max duration:", durations.max())
    # print("Mean duration:", durations.mean())
    # print("Median duration:", np.median(durations))
    # print("90th percentile:", np.percentile(durations, 90))
    # print("95th percentile:", np.percentile(durations, 95))