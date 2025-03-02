# Standard library imports
import os
import json
import shutil
import wave
import warnings
import logging
import subprocess
import sys
from pathlib import Path
from threading import Thread, Event
from queue import Queue
from datetime import datetime
from typing import Optional, Dict, List, Union, Iterator

# Third-party imports
import numpy as np
import librosa
import librosa.display  # Add this for better visualization
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Tkinter imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('media_converter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MediaConverterError(Exception):
    """Custom exception for media conversion errors"""
    pass

class AudioAnalyzer:
    """Handles advanced audio analysis and metadata"""
    def __init__(self):
        self.audio_info: Dict = {}
        self.logger = logging.getLogger(__name__ + '.AudioAnalyzer')
        
        # Suppress warnings
        # import warnings
        # warnings.filterwarnings("ignore", category=UserWarning)
        # warnings.filterwarnings("ignore", category=FutureWarning)

    def analyze_audio(self, file_path: Path) -> bool:
        """
        Perform detailed audio analysis
        Args:
            file_path: Path to audio file
        Returns:
            bool: True if analysis successful
        Raises:
            MediaConverterError: If analysis fails
        """
        try:
            self.logger.info(f"Analyzing audio file: {file_path}")
            
            # Use soundfile first, fall back to librosa
            try:
                import soundfile as sf
                data, sr = sf.read(str(file_path))
            except Exception:
                # If soundfile fails, use librosa with default parameters
                import librosa
                data, sr = librosa.load(str(file_path), sr=None)  # sr=None preserves original sample rate
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = librosa.to_mono(data)

            # Basic audio properties
            duration = librosa.get_duration(y=data, sr=sr)
            tempo, beats = librosa.beat.beat_track(y=data, sr=sr)
            
            # Audio features
            rmse = librosa.feature.rms(y=data)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(data)[0]
            
            # Silence detection with adjustable parameters
            intervals = librosa.effects.split(
                data, 
                top_db=20,  # Adjust this value for silence detection sensitivity
                frame_length=2048,
                hop_length=512
            )
            
            # Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(data)
            
            # Mel spectrogram with adjusted parameters
            mel_spec = librosa.feature.melspectrogram(
                y=data, 
                sr=sr,
                n_mels=128,
                fmax=sr/2
            )
            
            self.audio_info = {
                'duration': float(duration),
                'sample_rate': int(sr),
                'tempo': float(tempo),
                'beat_frames': beats.tolist(),
                'audio_features': {
                    'avg_volume': float(np.mean(rmse)),
                    'max_volume': float(np.max(rmse)),
                    'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                    'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                    'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                    'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                    'perceived_loudness': float(np.mean(librosa.power_to_db(rmse)))
                },
                'silent_intervals': intervals.tolist(),
                'components': {
                    'harmonic_mean': float(np.mean(np.abs(y_harmonic))),
                    'percussive_mean': float(np.mean(np.abs(y_percussive)))
                },
                'mel_spectrogram_mean': float(np.mean(mel_spec)),
                'file_info': {
                    'channels': 1 if len(data.shape) == 1 else data.shape[1],
                    'file_path': str(file_path),
                    'format': file_path.suffix[1:].upper()
                }
            }
            
            self.logger.info("Audio analysis completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Audio analysis failed: {str(e)}")

    def get_silence_regions(self) -> List[Dict[str, float]]:
        """
        Get silence regions with timestamps
        Returns:
            List of dicts containing start and end times of silence regions
        """
        if not self.audio_info.get('silent_intervals'):
            return []
            
        sr = self.audio_info['sample_rate']
        return [
            {
                'start': float(start) / sr,
                'end': float(end) / sr
            }
            for start, end in self.audio_info['silent_intervals']
        ]

    def export_analysis(self, filepath: Path) -> None:
        """
        Export analysis data to JSON file
        Args:
            filepath: Path to save JSON file
        Raises:
            MediaConverterError: If export fails
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.audio_info, f, indent=2)
            self.logger.info(f"Analysis exported to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export analysis: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Failed to export analysis: {str(e)}")

class WaveformVisualizer:
    """Base waveform visualization"""
    def __init__(self, canvas_frame):
        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        # Change from pack to grid
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.logger = logging.getLogger(__name__ + '.WaveformVisualizer')
        
    def plot_waveform(self, waveform_data: np.ndarray, sample_rate: int) -> None:
        """Plot audio waveform"""
        try:
            self.plot.clear()
            time = np.arange(len(waveform_data)) / sample_rate
            
            if len(waveform_data.shape) == 2:  # Stereo
                self.plot.plot(time, waveform_data[:, 0], 'b-', alpha=0.5, label='Left')
                self.plot.plot(time, waveform_data[:, 1], 'r-', alpha=0.5, label='Right')
                self.plot.legend()
            else:  # Mono
                self.plot.plot(time, waveform_data, 'g-')
                
            self.plot.set_xlabel('Time (seconds)')
            self.plot.set_ylabel('Amplitude')
            self.plot.grid(True)
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Failed to plot waveform: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Failed to plot waveform: {str(e)}")

    def update_selection(self, start_time: float, end_time: float) -> None:
        """Update the selection region on the waveform"""
        try:
            self.plot.axvline(x=start_time, color='g', linestyle='--')
            self.plot.axvline(x=end_time, color='r', linestyle='--')
            self.canvas.draw()
        except Exception as e:
            self.logger.error(f"Failed to update selection: {str(e)}", exc_info=True)

class EnhancedWaveformVisualizer(WaveformVisualizer):
    """Enhanced waveform visualization with interactive features"""
    def __init__(self, canvas_frame: ttk.Frame):
        super().__init__(canvas_frame)
        self.setup_interactive_elements()
        self.logger = logging.getLogger(__name__ + '.EnhancedWaveformVisualizer')
        
    def setup_interactive_elements(self) -> None:
        """Setup interactive visualization controls"""
        self.zoom_level = 1.0
        self.view_start = 0
        self.view_end = 1.0
        self.duration = 0
        
        # Enable mouse wheel zoom
        self.canvas.get_tk_widget().bind('<MouseWheel>', self.handle_zoom)
        # Enable drag to pan
        self.canvas.get_tk_widget().bind('<B1-Motion>', self.handle_pan)
        
        # Setup selection span
        self.span = SpanSelector(
            self.plot, self.on_select, 'horizontal',
            useblit=True, props=dict(alpha=0.5, facecolor='tab:blue')
        )
        
    def handle_zoom(self, event) -> None:
        """Handle mouse wheel zoom"""
        try:
            if event.delta > 0:
                self.zoom_level *= 1.1
            else:
                self.zoom_level /= 1.1
            self.zoom_level = np.clip(self.zoom_level, 1.0, 50.0)
            self.update_view()
        except Exception as e:
            self.logger.error(f"Zoom operation failed: {str(e)}", exc_info=True)
            
    def handle_pan(self, event) -> None:
        """Handle mouse drag panning"""
        try:
            if hasattr(self, 'last_x'):
                dx = event.x - self.last_x
                delta = dx * (self.view_end - self.view_start) / self.canvas.get_tk_widget().winfo_width()
                self.view_start = max(0, self.view_start - delta)
                self.view_end = min(1.0, self.view_end - delta)
                self.update_view()
            self.last_x = event.x
        except Exception as e:
            self.logger.error(f"Pan operation failed: {str(e)}", exc_info=True)
            
    def update_view(self) -> None:
        """Update the visible portion of the waveform"""
        try:
            self.plot.set_xlim(self.view_start * self.duration, 
                             self.view_end * self.duration)
            self.canvas.draw()
        except Exception as e:
            self.logger.error(f"View update failed: {str(e)}", exc_info=True)
            
    def on_select(self, xmin: float, xmax: float) -> None:
        """Handle selection on waveform"""
        try:
            self.selection_callback(xmin, xmax)
        except Exception as e:
            self.logger.error(f"Selection handling failed: {str(e)}", exc_info=True)

class AudioTrimmer:
    """Handles audio trimming operations"""
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
        self.waveform_data = None
        self.channels = 0
        self.sample_width = 0
        self.sample_rate = 0
        self.n_frames = 0
        self.logger = logging.getLogger(__name__ + '.AudioTrimmer')

    def load_audio_file(self, file_path: Path) -> bool:
        """
        Load and analyze audio file
        Args:
            file_path: Path to audio file
        Returns:
            bool: True if loading successful
        Raises:
            MediaConverterError: If loading fails
        """
        try:
            self.logger.info(f"Loading audio file: {file_path}")
            
            # Use different loading methods based on file extension
            suffix = file_path.suffix.lower()
            if suffix == '.wav':
                return self._load_wav_file(file_path)
            elif suffix in ['.mp3', '.m4a', '.aac', '.flac', '.ogg']:
                return self._load_pydub_file(file_path)
            else:
                return self._load_generic_audio(file_path)
                
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Error loading audio file: {str(e)}")

    def _load_pydub_file(self, file_path: Path) -> bool:
        """Load audio file using pydub"""
        from pydub import AudioSegment
        try:
            # Load audio file
            audio = AudioSegment.from_file(str(file_path))
            
            # Get audio properties
            self.channels = audio.channels
            self.sample_width = audio.sample_width
            self.sample_rate = audio.frame_rate
            self.duration = len(audio) / 1000.0  # Convert ms to seconds
            
            # Convert to numpy array
            samples = audio.get_array_of_samples()
            self.waveform_data = np.array(samples)
            
            # Reshape if stereo
            if self.channels == 2:
                self.waveform_data = self.waveform_data.reshape(-1, 2)
            
            # Normalize to float between -1 and 1
            self.waveform_data = self.waveform_data.astype(np.float32)
            max_val = float(1 << (8 * self.sample_width - 1))
            self.waveform_data /= max_val
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pydub loading failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Failed to load audio with pydub: {str(e)}")

    def _load_wav_file(self, file_path: Path) -> bool:
        """Load WAV file using wave module"""
        with wave.open(str(file_path), 'rb') as wav_file:
            self.channels = wav_file.getnchannels()
            self.sample_width = wav_file.getsampwidth()
            self.sample_rate = wav_file.getframerate()
            self.n_frames = wav_file.getnframes()
            self.duration = self.n_frames / float(self.sample_rate)
            
            frames = wav_file.readframes(self.n_frames)
            self.waveform_data = np.frombuffer(frames, dtype=np.int16)
            
            if self.channels == 2:
                self.waveform_data = self.waveform_data.reshape(-1, 2)
            
            # Normalize to float between -1 and 1
            self.waveform_data = self.waveform_data.astype(np.float32) / 32768.0
            
            return True

    def _load_generic_audio(self, file_path: Path) -> bool:
        """Load non-WAV audio files using soundfile"""
        data, samplerate = sf.read(str(file_path))
        self.waveform_data = data
        self.sample_rate = samplerate
        self.channels = 1 if len(data.shape) == 1 else data.shape[1]
        self.duration = len(data) / float(samplerate)
        return True



    def trim_audio(self, input_path: Path, output_path: Path, 
                  start_time: float, end_time: float, 
                  fade_duration: float = 0.1) -> bool:
        """
        Trim audio file with optional fade in/out
        
        Args:
            input_path: Source audio file path
            output_path: Output file path
            start_time: Start time in seconds
            end_time: End time in seconds
            fade_duration: Duration of fade in/out in seconds
            
        Returns:
            bool: True if trimming successful
            
        Raises:
            MediaConverterError: If trimming fails
        """
        try:
            self.logger.info(f"Trimming audio: {input_path} -> {output_path}")
            
            # Build FFmpeg command with fade
            command = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start_time),
                '-to', str(end_time),
                '-af', f'afade=t=in:st={start_time}:d={fade_duration},'
                      f'afade=t=out:st={end_time-fade_duration}:d={fade_duration}',
                '-y',
                str(output_path)
            ]
            
            # Run FFmpeg
            process = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg error: {process.stderr}")
                
            self.logger.info("Trimming completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Trimming failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Trimming failed: {str(e)}")

    def get_audio_info(self) -> Dict:
        """Get current audio file information"""
        return {
            'duration': self.duration,
            'channels': self.channels,
            'sample_rate': self.sample_rate,
            'sample_width': self.sample_width,
            'n_frames': self.n_frames
        }

class BatchProcessor:
    """Handles batch processing of audio files"""
    def __init__(self, converter):
        self.converter = converter
        self.batch_settings: Dict[Path, Dict] = {}
        self.queue = Queue()
        self.stop_event = Event()
        self.logger = logging.getLogger(__name__ + '.BatchProcessor')
        
    def add_to_batch(self, file_path: Path, settings: Dict) -> None:
        """
        Add file to batch processing queue
        
        Args:
            file_path: Audio file path
            settings: Processing settings dictionary
        """
        self.batch_settings[file_path] = settings
        self.logger.info(f"Added to batch: {file_path}")
        
    def remove_from_batch(self, file_path: Path) -> None:
        """Remove file from batch queue"""
        if file_path in self.batch_settings:
            del self.batch_settings[file_path]
            self.logger.info(f"Removed from batch: {file_path}")
            
    def clear_batch(self) -> None:
        """Clear all files from batch queue"""
        self.batch_settings.clear()
        self.logger.info("Batch queue cleared")
        
    def process_batch(self) -> None:
        """
        Process all files in batch
        
        Raises:
            MediaConverterError: If batch processing fails
        """
        try:
            self.logger.info("Starting batch processing")
            self.stop_event.clear()
            total = len(self.batch_settings)
            
            for i, (file_path, settings) in enumerate(self.batch_settings.items(), 1):
                if self.stop_event.is_set():
                    self.logger.info("Batch processing cancelled")
                    self.queue.put(('status', "Batch processing cancelled"))
                    break
                    
                try:
                    self.logger.info(f"Processing file {i}/{total}: {file_path}")
                    self.queue.put(('status', f"Processing: {file_path.name}"))
                    
                    # Process file with current settings
                    self.converter.process_file(file_path, settings)
                    
                    # Update progress
                    progress = (i / total) * 100
                    self.queue.put(('progress', progress))
                    self.queue.put(('log', f"Completed: {file_path.name}"))
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}", 
                                    exc_info=True)
                    self.queue.put(('error', f"Error processing {file_path}: {str(e)}"))
                    continue
                    
            if not self.stop_event.is_set():
                self.logger.info("Batch processing completed")
                self.queue.put(('status', "Batch processing completed"))
                
            self.queue.put(('done', None))
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
            self.queue.put(('error', f"Batch processing failed: {str(e)}"))
            self.queue.put(('done', None))
            
    def stop_processing(self) -> None:
        """Stop batch processing"""
        self.stop_event.set()
        self.logger.info("Batch processing stop requested")

    def get_batch_size(self) -> int:
        """Get number of files in batch queue"""
        return len(self.batch_settings)

    def get_batch_files(self) -> List[Path]:
        """Get list of files in batch queue"""
        return list(self.batch_settings.keys())

    def get_batch_settings(self, file_path: Path) -> Optional[Dict]:
        """Get settings for specific file in batch"""
        return self.batch_settings.get(file_path)

    def estimate_total_time(self) -> float:
        """
        Estimate total processing time for batch
        
        Returns:
            float: Estimated time in seconds
        """
        total_duration = 0
        for file_path, settings in self.batch_settings.items():
            try:
                with wave.open(str(file_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    total_duration += duration
            except Exception:
                # If can't read duration, estimate 5 minutes
                total_duration += 300
                
        # Estimate processing time (assuming processing is 2x realtime)
        return total_duration * 2

class MediaConverter:
    """Main application class for media conversion and audio processing"""
    def __init__(self):
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.current_audio_file: Optional[Path] = None
        self.stop_event = Event()
        
        # Initialize components
        self.analyzer = AudioAnalyzer()
        self.trimmer = AudioTrimmer()
        self.batch_processor = BatchProcessor(self)
        
        # Format options
        self.format_options = {
            'WAV': {
                'extension': 'wav',
                'codec': 'pcm_s16le',
                'bitrates': ['1411k'],
                'sample_rates': ['44100', '48000', '96000']
            },
            'MP3': {
                'extension': 'mp3',
                'codec': 'libmp3lame',
                'bitrates': ['320k', '256k', '192k', '128k'],
                'sample_rates': ['44100', '48000']
            },
            'AAC': {
                'extension': 'm4a',
                'codec': 'aac',
                'bitrates': ['256k', '192k', '128k'],
                'sample_rates': ['44100', '48000']
            },
            'FLAC': {
                'extension': 'flac',
                'codec': 'flac',
                'bitrates': ['0'],  # Lossless
                'sample_rates': ['44100', '48000', '96000']
            }
        }
        
        # Default values
        self.current_format = 'WAV'
        self.current_bitrate = '320k'
        self.current_sample_rate = '44100'
        
        self.logger = logging.getLogger(__name__ + '.MediaConverter')
        self.verify_ffmpeg()
        self.setup_ui()

    def verify_ffmpeg(self) -> None:
        """
        Verify FFmpeg installation
        
        Raises:
            MediaConverterError: If FFmpeg is not installed
        """
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise MediaConverterError("FFmpeg not found")
            self.logger.info("FFmpeg verification successful")
        except FileNotFoundError:
            self.logger.error("FFmpeg not found in system PATH")
            raise MediaConverterError(
                "FFmpeg is not installed or not in system PATH")

    def setup_ui(self) -> None:
        """Initialize the main UI window"""
        self.root = tk.Tk()
        self.root.title("Enhanced Media Converter")
        self.root.geometry("800x900")
        
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Setup UI sections
        self.setup_format_section(main_container)
        self.setup_path_section(main_container)
        self.setup_trim_section(main_container)
        self.setup_analysis_section(main_container)
        self.setup_batch_section(main_container)
        self.setup_progress_section(main_container)
        self.setup_log_section(main_container)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize queue for thread communication
        self.queue = Queue()

    def setup_format_section(self, parent: ttk.Frame) -> None:
        """Setup format selection section"""
        format_frame = ttk.LabelFrame(parent, text="Output Format", padding="5")
        format_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Format selection
        ttk.Label(format_frame, text="Format:").grid(row=0, column=0, sticky=tk.W)
        self.format_var = tk.StringVar(value=self.current_format)
        format_combo = ttk.Combobox(
            format_frame, 
            textvariable=self.format_var,
            values=list(self.format_options.keys())
        )
        format_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        format_combo.bind('<<ComboboxSelected>>', self.update_format_options)
        
        # Bitrate selection
        ttk.Label(format_frame, text="Bitrate:").grid(row=1, column=0, sticky=tk.W)
        self.bitrate_var = tk.StringVar(value=self.current_bitrate)
        self.bitrate_combo = ttk.Combobox(format_frame, textvariable=self.bitrate_var)
        self.bitrate_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Sample rate selection
        ttk.Label(format_frame, text="Sample Rate:").grid(row=2, column=0, sticky=tk.W)
        self.sample_rate_var = tk.StringVar(value=self.current_sample_rate)
        self.sample_rate_combo = ttk.Combobox(
            format_frame, 
            textvariable=self.sample_rate_var
        )
        self.sample_rate_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Channel configuration
        ttk.Label(format_frame, text="Channels:").grid(row=3, column=0, sticky=tk.W)
        self.channel_var = tk.StringVar(value="2")
        channel_combo = ttk.Combobox(
            format_frame,
            textvariable=self.channel_var,
            values=["1", "2"]
        )
        channel_combo.grid(row=3, column=1, sticky=tk.W, padx=5)
        
        self.update_format_options()

    def setup_path_section(self, parent: ttk.Frame) -> None:
        """Setup file path selection section"""
        path_frame = ttk.LabelFrame(parent, text="File Selection", padding="5")
        path_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Input path
        input_frame = ttk.Frame(path_frame)
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(input_frame, text="Input:").grid(row=0, column=0, sticky=tk.W)
        self.input_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path_var, 
                 width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Browse", 
                  command=self.select_input).grid(row=0, column=2)
        
        # Output path
        output_frame = ttk.Frame(path_frame)
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(output_frame, text="Output:").grid(row=0, column=0, sticky=tk.W)
        self.output_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path_var, 
                 width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="Browse", 
                  command=self.select_output).grid(row=0, column=2)

    def setup_trim_section(self, parent: ttk.Frame) -> None:
        """Setup audio trimming section"""
        trim_frame = ttk.LabelFrame(parent, text="Audio Trimming", padding="5")
        trim_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Mode selection
        mode_frame = ttk.Frame(trim_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.trim_mode = tk.StringVar(value="simple")
        ttk.Radiobutton(mode_frame, text="Simple", variable=self.trim_mode,
                    value="simple", command=self.toggle_trim_mode).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Advanced", variable=self.trim_mode,
                    value="advanced", command=self.toggle_trim_mode).grid(row=0, column=1, padx=5)
        
        # Simple mode frame
        self.simple_trim_frame = ttk.Frame(trim_frame)
        self.simple_trim_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Time inputs for simple mode
        time_frame = ttk.Frame(self.simple_trim_frame)
        time_frame.grid(row=0, column=0, pady=5)
        
        ttk.Label(time_frame, text="Start Time:").grid(row=0, column=0)
        self.start_time_var = tk.StringVar(value="00:00:00")
        ttk.Entry(time_frame, textvariable=self.start_time_var, 
                width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(time_frame, text="End Time:").grid(row=0, column=2)
        self.end_time_var = tk.StringVar(value="00:00:00")
        ttk.Entry(time_frame, textvariable=self.end_time_var, 
                width=10).grid(row=0, column=3, padx=5)
        
        # Advanced mode frame
        self.advanced_trim_frame = ttk.Frame(trim_frame)
        self.advanced_trim_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Create waveform frame
        waveform_frame = ttk.Frame(self.advanced_trim_frame)
        waveform_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Initialize waveform visualizer
        self.waveform_visualizer = EnhancedWaveformVisualizer(waveform_frame)
        self.waveform_visualizer.selection_callback = self.handle_waveform_selection
        
        # Timeline slider
        self.timeline_var = tk.DoubleVar(value=0)
        self.timeline = Scale(
            self.advanced_trim_frame,
            from_=0, to=100,
            orient=tk.HORIZONTAL,
            variable=self.timeline_var,
            command=self.update_timeline
        )
        self.timeline.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Common controls for both modes
        control_frame = ttk.Frame(trim_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(control_frame, text="Preview", 
                command=self.preview_trim).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Apply Trim",
                command=self.apply_trim).grid(row=0, column=1, padx=5)
        
        # Presets
        preset_frame = ttk.Frame(trim_frame)
        preset_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Label(preset_frame, text="Presets:").grid(row=0, column=0, padx=5)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=["Remove Silence", "First 30 sec", "Last 30 sec", "Custom..."],
            width=20
        )
        preset_combo.grid(row=0, column=1, padx=5)
        preset_combo.bind('<<ComboboxSelected>>', self.apply_preset)
        
        # Initially show simple mode
        self.toggle_trim_mode()

    def apply_trim(self) -> None:
        """Apply trim settings to the current audio file"""
        try:
            if not self.current_audio_file:
                raise MediaConverterError("No audio file loaded")
                
            start = self.parse_time(self.start_time_var.get())
            end = self.parse_time(self.end_time_var.get())
            
            if start >= end:
                raise MediaConverterError("Start time must be less than end time")
                
            output_file = self.current_audio_file.parent / f"trimmed_{self.current_audio_file.name}"
            
            if self.trimmer.trim_audio(self.current_audio_file, output_file, start, end):
                self.log_message(f"Successfully trimmed: {output_file}")
                messagebox.showinfo("Success", "Trim operation completed successfully")
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def apply_preset(self, event=None) -> None:
        """Apply selected trimming preset"""
        try:
            if not self.current_audio_file:
                raise MediaConverterError("No audio file loaded")
                
            preset = self.preset_var.get()
            
            if preset == "Remove Silence":
                if not self.analyzer.audio_info:
                    self.analyze_current_file()
                silence_regions = self.analyzer.get_silence_regions()
                if silence_regions:
                    # Find longest non-silent section
                    max_duration = 0
                    start_time = 0
                    end_time = self.trimmer.duration
                    
                    for i in range(len(silence_regions) - 1):
                        duration = silence_regions[i + 1]['start'] - silence_regions[i]['end']
                        if duration > max_duration:
                            max_duration = duration
                            start_time = silence_regions[i]['end']
                            end_time = silence_regions[i + 1]['start']
                    
                    self.start_time_var.set(self.format_time(start_time))
                    self.end_time_var.set(self.format_time(end_time))
                    
            elif preset == "First 30 sec":
                self.start_time_var.set("00:00:00")
                self.end_time_var.set("00:00:30")
                
            elif preset == "Last 30 sec":
                if self.trimmer.duration > 30:
                    start = self.trimmer.duration - 30
                    self.start_time_var.set(self.format_time(start))
                    self.end_time_var.set(self.format_time(self.trimmer.duration))
            
            # Update waveform selection if in advanced mode
            if self.trim_mode.get() == "advanced":
                start = self.parse_time(self.start_time_var.get())
                end = self.parse_time(self.end_time_var.get())
                self.waveform_visualizer.update_selection(start, end)
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def setup_analysis_section(self, parent: ttk.Frame) -> None:
        """Setup audio analysis section"""
        analysis_frame = ttk.LabelFrame(parent, text="Audio Analysis", padding="5")
        analysis_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Analysis controls
        control_frame = ttk.Frame(analysis_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(control_frame, text="Analyze Audio",
                  command=self.analyze_current_file).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Export Analysis",
                  command=self.export_analysis).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Detect Silence",
                  command=self.detect_silence).grid(row=0, column=2, padx=5)
        
        # Analysis display
        self.analysis_text = tk.Text(analysis_frame, height=6, width=70)
        self.analysis_text.grid(row=1, column=0, pady=5)
        
        # Scrollbar for analysis text
        analysis_scroll = ttk.Scrollbar(analysis_frame, orient="vertical",
                                      command=self.analysis_text.yview)
        analysis_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.analysis_text.configure(yscrollcommand=analysis_scroll.set)

    def setup_batch_section(self, parent: ttk.Frame) -> None:
        """Setup batch processing section"""
        batch_frame = ttk.LabelFrame(parent, text="Batch Processing", padding="5")
        batch_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Batch controls
        control_frame = ttk.Frame(batch_frame)
        control_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(control_frame, text="Add to Batch",
                  command=self.add_to_batch).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Remove Selected",
                  command=self.remove_from_batch).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Clear Batch",
                  command=self.clear_batch).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Process Batch",
                  command=self.process_batch).grid(row=0, column=3, padx=5)
        
        # Batch queue display
        self.batch_list = tk.Listbox(batch_frame, height=5, width=70)
        self.batch_list.grid(row=1, column=0, pady=5)
        
        # Scrollbar for batch list
        batch_scroll = ttk.Scrollbar(batch_frame, orient="vertical",
                                   command=self.batch_list.yview)
        batch_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.batch_list.configure(yscrollcommand=batch_scroll.set)

    def setup_progress_section(self, parent: ttk.Frame) -> None:
        """Setup progress tracking section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        progress_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=600,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=2
        )
        
        # Control buttons
        button_frame = ttk.Frame(progress_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.convert_btn = ttk.Button(
            button_frame,
            text="Convert",
            command=self.start_conversion
        )
        self.convert_btn.grid(row=0, column=0, padx=5)
        
        self.cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_conversion,
            state='disabled'
        )
        self.cancel_btn.grid(row=0, column=1, padx=5)

    def setup_log_section(self, parent: ttk.Frame) -> None:
        """Setup logging section"""
        log_frame = ttk.LabelFrame(parent, text="Log", padding="5")
        log_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Log display
        self.log_text = tk.Text(log_frame, height=8, width=70)
        self.log_text.grid(row=0, column=0, pady=5)
        
        # Scrollbar for log
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical",
                                 command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        # Log controls
        control_frame = ttk.Frame(log_frame)
        control_frame.grid(row=1, column=0, columnspan=2)
        
        ttk.Button(control_frame, text="Save Log",
                  command=self.save_log).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Clear Log",
                  command=self.clear_log).grid(row=0, column=1, padx=5)

    def update_format_options(self, event=None) -> None:
        """Update available options based on selected format"""
        format_info = self.format_options[self.format_var.get()]
        
        # Update bitrate options
        self.bitrate_combo['values'] = format_info['bitrates']
        if self.bitrate_var.get() not in format_info['bitrates']:
            self.bitrate_var.set(format_info['bitrates'][0])
            
        # Update sample rate options
        self.sample_rate_combo['values'] = format_info['sample_rates']
        if self.sample_rate_var.get() not in format_info['sample_rates']:
            self.sample_rate_var.set(format_info['sample_rates'][0])

    def toggle_trim_mode(self) -> None:
        """Switch between simple and advanced trimming modes"""
        if self.trim_mode.get() == "simple":
            self.advanced_trim_frame.grid_remove()
            self.simple_trim_frame.grid()
        else:
            self.simple_trim_frame.grid_remove()
            self.advanced_trim_frame.grid()
            if self.current_audio_file:
                self.load_waveform()

    def handle_waveform_selection(self, start_time: float, end_time: float) -> None:
        """Handle selection made on waveform"""
        self.start_time_var.set(self.format_time(start_time))
        self.end_time_var.set(self.format_time(end_time))
        self.waveform_visualizer.update_selection(start_time, end_time)

    def update_timeline(self, value: str) -> None:
        """Update timeline position"""
        if hasattr(self, 'waveform_visualizer'):
            position = float(value) / 100 * self.trimmer.duration
            self.waveform_visualizer.plot.axvline(
                x=position,
                color='g',
                linestyle='--',
                alpha=0.5
            )
            self.waveform_visualizer.canvas.draw()

    def format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def parse_time(self, time_str: str) -> float:
        """Convert time string to seconds"""
        try:
            if not time_str or time_str.strip() == "":
                return 0.0

            # Handle various time formats
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
            elif len(parts) == 1:
                return float(parts[0])
            else:
                return 0.0
        except:
            self.logger.warning(f"Invalid time format: {time_str}, using 0.0")
            return 0.0

    def load_waveform(self) -> None:
        """Load and display audio waveform"""
        try:
            if self.trimmer.load_audio_file(self.current_audio_file):
                self.waveform_visualizer.plot_waveform(
                    self.trimmer.waveform_data,
                    self.trimmer.sample_rate
                )
                self.timeline.configure(to=self.trimmer.duration)
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def get_file_duration(self, file_path: Path) -> float:
        """Get duration of audio file using FFmpeg"""
        try:
            command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"Could not get file duration: {e}")
        return 0.0

    def select_input(self) -> None:
        """Open dialog for input selection"""
        try:
            choice = messagebox.askquestion(
                "Input Selection",
                "Would you like to select a single file?\n\n" +
                "Select 'Yes' for single file\n" +
                "Select 'No' for entire folder"
            )
            
            if choice == 'yes':
                path = filedialog.askopenfilename(
                    title="Select media file",
                    filetypes=[
                        ("Audio files", "*.wav *.mp3 *.aac *.m4a *.flac *.ogg"),
                        ("All files", "*.*")
                    ]
                )
            else:
                path = filedialog.askdirectory(title="Select input folder")
                
            if path:
                self.input_path = Path(path)
                self.input_path_var.set(str(self.input_path))
                self.current_audio_file = self.input_path
                self.validate_input_path()
                
                # Get file duration and update end time
                if self.input_path.is_file():
                    duration = self.get_file_duration(self.input_path)
                    if duration > 0:
                        self.end_time_var.set(self.format_time(duration))
                
                self.log_message(f"Selected input: {self.input_path}")
                
                if self.trim_mode.get() == "advanced":
                    self.load_waveform()
                    
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))
            self.input_path = None
            self.input_path_var.set("")
            
    def select_output(self) -> None:
        """Open dialog for output folder selection"""
        try:
            path = filedialog.askdirectory(title="Select output folder")
            if path:
                self.output_path = Path(path)
                self.output_path_var.set(str(self.output_path))
                self.validate_output_path()
                self.log_message(f"Selected output: {self.output_path}")
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))
            self.output_path = None
            self.output_path_var.set("")

    def validate_input_path(self) -> None:
        """Validate input path exists and contains media files"""
        if not self.input_path.exists():
            raise MediaConverterError(f"Path does not exist: {self.input_path}")
            
        if self.input_path.is_file():
            if not self.is_media_file(self.input_path):
                raise MediaConverterError(
                    f"File is not a supported media format: {self.input_path}")
        else:
            media_files = list(self.get_media_files(self.input_path))
            if not media_files:
                raise MediaConverterError(
                    f"No supported media files found in: {self.input_path}")

    def validate_output_path(self) -> None:
        """Validate output path is writable"""
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            test_file = self.output_path / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise MediaConverterError(
                f"Cannot write to output directory: {self.output_path}\nError: {str(e)}")

    def is_media_file(self, path: Path) -> bool:
        """Check if file is a supported media format"""
        return path.suffix.lower() in (
            '.wav', '.mp3', '.aac', '.m4a', '.flac', '.ogg'
        )

    def get_media_files(self, directory: Path) -> Iterator[Path]:
        """Generator for supported media files in directory"""
        for path in directory.glob('*'):
            if self.is_media_file(path):
                yield path

    def start_conversion(self) -> None:
        """Start the conversion process"""
        try:
            if not self.input_path or not self.output_path:
                raise MediaConverterError("Please select input and output paths")
                
            self.convert_btn.state(['disabled'])
            self.cancel_btn.state(['!disabled'])
            self.progress_var.set(0)
            self.status_var.set("Starting conversion...")
            
            # Start conversion thread
            Thread(target=self.convert_files, daemon=True).start()
            self.root.after(100, self.check_queue)
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def convert_files(self) -> None:
        """Process all files in input path"""
        try:
            # Reset stop event
            self.stop_event.clear()
            
            # Get files to convert
            files_to_convert = ([self.input_path] if self.input_path.is_file() 
                              else list(self.get_media_files(self.input_path)))
            
            # Process each file
            total_files = len(files_to_convert)
            for i, file_path in enumerate(files_to_convert, 1):
                if self.stop_event.is_set():
                    self.queue.put(('status', "Conversion cancelled"))
                    break
                    
                self.queue.put(('progress', (i / total_files) * 100))
                self.queue.put(('status', f"Converting {file_path.name}"))
                
                try:
                    self.process_file(file_path)
                    self.queue.put(('log', f"Successfully converted {file_path.name}"))
                except Exception as e:
                    self.queue.put(('log', f"Error converting {file_path.name}: {str(e)}"))
                    continue
                    
            if not self.stop_event.is_set():
                self.queue.put(('status', "Conversion complete"))
            self.queue.put(('done', None))
            
        except Exception as e:
            self.queue.put(('error', str(e)))
            self.queue.put(('done', None))

    def process_file(self, file_path: Path, settings: Optional[Dict] = None) -> None:
        """
        Process a single file with conversion and trimming
        Args:
            file_path: Path to input file
            settings: Optional dictionary of processing settings
        """
        try:
            # Use provided settings or current UI settings
            if settings is None:
                settings = {
                    'start_time': self.start_time_var.get(),
                    'end_time': self.end_time_var.get(),
                    'format': self.format_var.get(),
                    'bitrate': self.bitrate_var.get(),
                    'sample_rate': self.sample_rate_var.get(),
                    'channels': self.channel_var.get()
                }

            format_info = self.format_options[settings['format']]
            output_file = self.output_path / f"{file_path.stem}.{format_info['extension']}"

            # Get file duration
            file_duration = self.get_file_duration(file_path)

            # Build FFmpeg command
            command = [
                'ffmpeg',
                '-i', str(file_path)
            ]

            # Add trim parameters if needed
            start_time = self.parse_time(settings['start_time'])
            end_time = self.parse_time(settings['end_time'])
            
            # Validate trim times
            if start_time >= file_duration:
                start_time = 0
            if end_time <= start_time or end_time > file_duration:
                end_time = file_duration

            # Only add trim parameters if they're valid
            if start_time > 0 or end_time < file_duration:
                command.extend([
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time)  # Use duration instead of end point
                ])

            # Add conversion parameters
            command.extend([
                '-acodec', format_info['codec'],
                '-ar', settings['sample_rate'],
                '-ac', settings['channels'],
                '-b:a', settings['bitrate'],
                '-y',  # Overwrite output
                str(output_file)
            ])

            # Log the command for debugging
            self.logger.info(f"FFmpeg command: {' '.join(command)}")

            # Run conversion
            process = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg error: {process.stderr}")

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Processing failed: {str(e)}")

    def cancel_conversion(self) -> None:
        """Cancel the ongoing conversion process"""
        self.stop_event.set()
        self.cancel_btn.state(['disabled'])
        self.status_var.set("Cancelling...")

    def check_queue(self) -> None:
        """Check for updates from the conversion thread"""
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == 'progress':
                    self.progress_var.set(data)
                elif msg_type == 'status':
                    self.status_var.set(data)
                elif msg_type == 'log':
                    self.log_message(data)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    self.convert_btn.state(['!disabled'])
                    self.cancel_btn.state(['disabled'])
                elif msg_type == 'done':
                    self.convert_btn.state(['!disabled'])
                    self.cancel_btn.state(['disabled'])
                    break
                    
        except Exception:
            # Queue is empty, schedule next check
            self.root.after(100, self.check_queue)

    def analyze_current_file(self) -> None:
        """Analyze current audio file"""
        try:
            if not self.current_audio_file:
                raise MediaConverterError("No audio file loaded")
                
            self.status_var.set("Analyzing audio...")
            if self.analyzer.analyze_audio(self.current_audio_file):
                self.display_analysis()
                self.status_var.set("Analysis complete")
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Analysis failed")

    def display_analysis(self) -> None:
        """Display audio analysis results"""
        self.analysis_text.delete(1.0, tk.END)
        
        info = self.analyzer.audio_info
        display_text = f"""
Audio Analysis Results:
----------------------
Duration: {info['duration']:.2f} seconds
Sample Rate: {info['sample_rate']} Hz
Tempo: {info['tempo']:.2f} BPM

Audio Features:
--------------
Average Volume: {info['audio_features']['avg_volume']:.2f}
Max Volume: {info['audio_features']['max_volume']:.2f}
Spectral Centroid: {info['audio_features']['spectral_centroid_mean']:.2f}
Zero Crossing Rate: {info['audio_features']['zero_crossing_rate_mean']:.2f}

Silent Intervals: {len(info.get('silent_intervals', []))}
        """
        
        self.analysis_text.insert(tk.END, display_text)

    def preview_trim(self) -> None:
        """Preview the trimmed audio"""
        try:
            if not self.current_audio_file:
                raise MediaConverterError("No audio file loaded")
                
            start = self.parse_time(self.start_time_var.get())
            end = self.parse_time(self.end_time_var.get())
            
            # Create temporary file for preview
            temp_output = self.current_audio_file.parent / f"preview_{self.current_audio_file.name}"
            
            if self.trimmer.trim_audio(self.current_audio_file, temp_output, start, end):
                # Play preview using ffplay
                subprocess.Popen(['ffplay', '-nodisp', '-autoexit', str(temp_output)])
                self.log_message("Playing preview...")
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def add_to_batch(self) -> None:
        """Add current file and settings to batch"""
        try:
            if not self.current_audio_file:
                raise MediaConverterError("No audio file loaded")
                
            settings = {
                'start_time': self.start_time_var.get(),
                'end_time': self.end_time_var.get(),
                'format': self.format_var.get(),
                'bitrate': self.bitrate_var.get(),
                'sample_rate': self.sample_rate_var.get(),
                'channels': self.channel_var.get()
            }
            
            self.batch_processor.add_to_batch(self.current_audio_file, settings)
            self.batch_list.insert(tk.END, self.current_audio_file.name)
            self.log_message(f"Added to batch: {self.current_audio_file.name}")
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def remove_from_batch(self) -> None:
        """Remove selected file from batch queue"""
        selection = self.batch_list.curselection()
        if selection:
            index = selection[0]
            filename = self.batch_list.get(index)
            file_path = self.input_path.parent / filename
            
            self.batch_processor.remove_from_batch(file_path)
            self.batch_list.delete(index)
            self.log_message(f"Removed from batch: {filename}")

    def clear_batch(self) -> None:
        """Clear all files from batch queue"""
        self.batch_processor.clear_batch()
        self.batch_list.delete(0, tk.END)
        self.log_message("Batch queue cleared")

    def process_batch(self) -> None:
        """Process all files in batch queue"""
        try:
            if self.batch_list.size() == 0:
                raise MediaConverterError("Batch queue is empty")
                
            if not self.output_path:
                raise MediaConverterError("Please select output directory")
                
            self.convert_btn.state(['disabled'])
            self.cancel_btn.state(['!disabled'])
            self.progress_var.set(0)
            self.status_var.set("Starting batch processing...")
            
            # Estimate total time
            estimated_time = self.batch_processor.estimate_total_time()
            self.log_message(f"Estimated processing time: {estimated_time:.1f} seconds")
            
            # Start processing thread
            Thread(target=self.batch_processor.process_batch, daemon=True).start()
            self.root.after(100, self.check_batch_queue)
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def check_batch_queue(self) -> None:
        """Check batch processing progress"""
        try:
            while True:
                msg_type, data = self.batch_processor.queue.get_nowait()
                
                if msg_type == 'progress':
                    self.progress_var.set(data)
                elif msg_type == 'status':
                    self.status_var.set(data)
                elif msg_type == 'log':
                    self.log_message(data)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                elif msg_type == 'done':
                    self.convert_btn.state(['!disabled'])
                    self.cancel_btn.state(['disabled'])
                    break
                    
        except Exception:
            self.root.after(100, self.check_batch_queue)

    def log_message(self, message: str) -> None:
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.logger.info(message)

    def save_log(self) -> None:
        """Save log contents to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"conversion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"Log saved to: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {str(e)}")

    def clear_log(self) -> None:
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared")

    def export_analysis(self) -> None:
        """Export audio analysis to JSON"""
        try:
            if not self.analyzer.audio_info:
                raise MediaConverterError("No analysis data available")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialfile=f"audio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if file_path:
                self.analyzer.export_analysis(Path(file_path))
                self.log_message(f"Analysis exported to: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def detect_silence(self) -> None:
        """Detect and mark silent intervals"""
        try:
            if not self.analyzer.audio_info:
                raise MediaConverterError("No analysis data available")
                
            silence_regions = self.analyzer.get_silence_regions()
            
            if not silence_regions:
                self.log_message("No silent regions detected")
                return
                
            # Update analysis display
            self.analysis_text.insert(tk.END, "\nSilent Regions:\n--------------\n")
            for i, region in enumerate(silence_regions, 1):
                self.analysis_text.insert(
                    tk.END,
                    f"Region {i}: {self.format_time(region['start'])} - "
                    f"{self.format_time(region['end'])}\n"
                )
            
            # Mark silent regions on waveform if in advanced mode
            if self.trim_mode.get() == "advanced":
                for region in silence_regions:
                    self.waveform_visualizer.plot.axvspan(
                        region['start'],
                        region['end'],
                        alpha=0.3,
                        color='red'
                    )
                self.waveform_visualizer.canvas.draw()
                
            self.log_message(f"Detected {len(silence_regions)} silent regions")
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

def main():
    """Main application entry point"""
    try:
        # Setup logging
        log_file = Path('media_converter.log')
        if log_file.exists():
            # Backup old log file
            backup_name = f"media_converter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_file.rename(log_file.parent / backup_name)
            
        # Initialize application
        converter = MediaConverter()
        
        # Set window icon if available
        try:
            if sys.platform == 'win32':
                converter.root.iconbitmap('icon.ico')
        except Exception:
            pass
            
        # Start application
        converter.root.mainloop()
        
    except MediaConverterError as e:
        messagebox.showerror("Startup Error", str(e))
        logging.error(f"Startup error: {str(e)}", exc_info=True)
    except Exception as e:
        messagebox.showerror(
            "Startup Error",
            f"Unexpected error during startup: {str(e)}"
        )
        logging.error(f"Unexpected startup error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()