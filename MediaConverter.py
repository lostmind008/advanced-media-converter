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
from typing import Optional, Dict, List, Union, Iterator, Tuple

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
from tkinter import ttk, filedialog, messagebox, Scale, IntVar, BooleanVar, Checkbutton

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


# Fix for the librosa import error in analyze_audio method
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
            
            # Import libraries locally to ensure they're available in this context
            import librosa
            import numpy as np
            
            # Use soundfile first, fall back to librosa
            try:
                import soundfile as sf
                data, sr = sf.read(str(file_path))
            except Exception:
                # If soundfile fails, use librosa with default parameters
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
        self.selections = []  # Store multiple selections
        
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
    
    def add_selection(self, start_time: float, end_time: float) -> None:
        """Add a selection to the list of selections"""
        self.selections.append((start_time, end_time))
        # Highlight this selection on the waveform
        self.plot.axvspan(start_time, end_time, alpha=0.2, color='blue')
        self.canvas.draw()
    
    def clear_selections(self) -> None:
        """Clear all selections"""
        self.selections = []
        self.plot.clear()
        # Redraw the waveform
        if hasattr(self, 'last_waveform_data') and hasattr(self, 'last_sample_rate'):
            self.plot_waveform(self.last_waveform_data, self.last_sample_rate)

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
        self.selections = []  # Store multiple selections

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

    def trim_multiple_segments(self, input_path: Path, output_path: Path, 
                             segments: List[Tuple[float, float]], 
                             fade_duration: float = 0.1) -> bool:
        """
        Trim multiple segments and concatenate them into a single file
        
        Args:
            input_path: Source file path
            output_path: Output file path
            segments: List of (start_time, end_time) tuples in seconds
            fade_duration: Duration of fade in/out in seconds
            
        Returns:
            bool: True if successful
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Trimming multiple segments from: {input_path}")
            
            if not segments:
                raise MediaConverterError("No segments provided for trimming")
                
            # Create temporary directory for segments
            temp_dir = Path(os.path.dirname(output_path)) / "temp_segments"
            temp_dir.mkdir(exist_ok=True)
            
            # Create a file to list all segments for concatenation
            concat_file = temp_dir / "concat_list.txt"
            
            # Extract each segment
            segment_files = []
            
            for i, (start_time, end_time) in enumerate(segments):
                segment_path = temp_dir / f"segment_{i}.ts"
                segment_files.append(segment_path)
                
                # Build FFmpeg command for each segment
                command = [
                    'ffmpeg',
                    '-i', str(input_path),
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-af', f'afade=t=in:st=0:d={fade_duration},'
                          f'afade=t=out:st={end_time-start_time-fade_duration}:d={fade_duration}',
                    '-c', 'copy',  # Try to copy codecs when possible for speed
                    '-avoid_negative_ts', '1',
                    '-y',
                    str(segment_path)
                ]
                
                # Run FFmpeg
                process = subprocess.run(command, capture_output=True, text=True)
                
                if process.returncode != 0:
                    raise MediaConverterError(f"FFmpeg segment error: {process.stderr}")
            
            # Create concatenation file
            with open(concat_file, 'w') as f:
                for segment in segment_files:
                    f.write(f"file '{segment}'\n")
            
            # Concatenate all segments
            concat_command = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',  # Try to copy codecs when possible
                '-y',
                str(output_path)
            ]
            
            process = subprocess.run(concat_command, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg concatenation error: {process.stderr}")
            
            # Clean up temporary files
            for segment in segment_files:
                if segment.exists():
                    segment.unlink()
            concat_file.unlink()
            temp_dir.rmdir()
                
            self.logger.info("Multiple segment trimming completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Multiple segment trimming failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Multiple segment trimming failed: {str(e)}")

    def get_audio_info(self) -> Dict:
        """Get current audio file information"""
        return {
            'duration': self.duration,
            'channels': self.channels,
            'sample_rate': self.sample_rate,
            'sample_width': self.sample_width,
            'n_frames': self.n_frames
        }

    def add_selection(self, start_time: float, end_time: float) -> None:
        """Add a selection to the list of selections"""
        self.selections.append((start_time, end_time))
        self.logger.info(f"Added selection {len(self.selections)}: {start_time:.2f}s to {end_time:.2f}s")
    
    def clear_selections(self) -> None:
        """Clear all selections"""
        self.logger.info(f"Cleared {len(self.selections)} selections")
        self.selections = []

class VideoProcessor:
    """Handles video processing operations"""
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.VideoProcessor')
        self.duration = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.video_codec = ""
        self.audio_codec = ""
        self.bitrate = ""
        self.metadata = {}
    
    def get_video_info(self, file_path: Path) -> Dict:
        """
        Get comprehensive video file information
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dict: Video properties
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Getting video info for: {file_path}")
            
            # Use FFprobe to get video info
            command = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise MediaConverterError(f"FFprobe error: {result.stderr}")
                
            info = json.loads(result.stdout)
            
            # Extract relevant information
            self.metadata = info
            
            # Get video stream info
            video_stream = None
            audio_stream = None
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                elif stream.get('codec_type') == 'audio':
                    audio_stream = stream
            
            if video_stream:
                self.width = int(video_stream.get('width', 0))
                self.height = int(video_stream.get('height', 0))
                # Calculate fps from frame rate fraction
                fps_frac = video_stream.get('r_frame_rate', '0/1')
                if '/' in fps_frac:
                    num, denom = map(int, fps_frac.split('/'))
                    self.fps = num / denom if denom else 0
                else:
                    self.fps = float(fps_frac)
                self.video_codec = video_stream.get('codec_name', '')
            
            if audio_stream:
                self.audio_codec = audio_stream.get('codec_name', '')
            
            # Get format info
            if 'format' in info:
                self.duration = float(info['format'].get('duration', 0))
                self.bitrate = info['format'].get('bit_rate', '')
            
            video_info = {
                'duration': self.duration,
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'video_codec': self.video_codec,
                'audio_codec': self.audio_codec,
                'bitrate': self.bitrate,
                'file_size': os.path.getsize(file_path),
                'format': info.get('format', {}).get('format_name', '')
            }
            
            self.logger.info(f"Video info retrieved successfully: {video_info}")
            return video_info
            
        except Exception as e:
            self.logger.error(f"Failed to get video info: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Failed to get video info: {str(e)}")

    def extract_frame(self, file_path: Path, time_pos: float, output_path: Path) -> bool:
        """
        Extract a frame from video at specified time
        
        Args:
            file_path: Path to video file
            time_pos: Time position in seconds
            output_path: Path to save the frame
            
        Returns:
            bool: True if successful
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Extracting frame at {time_pos}s from: {file_path}")
            
            # Build FFmpeg command
            command = [
                'ffmpeg',
                '-ss', str(time_pos),
                '-i', str(file_path),
                '-vframes', '1',
                '-q:v', '2',
                '-y',
                str(output_path)
            ]
            
            # Run FFmpeg
            process = subprocess.run(command, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg frame extraction error: {process.stderr}")
                
            self.logger.info(f"Frame extracted successfully to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Frame extraction failed: {str(e)}")

    def extract_audio(self, file_path: Path, output_path: Path) -> bool:
        """
        Extract audio track from video
        
        Args:
            file_path: Path to video file
            output_path: Path to save audio file
            
        Returns:
            bool: True if successful
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Extracting audio from: {file_path}")
            
            # Build FFmpeg command
            command = [
                'ffmpeg',
                '-i', str(file_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # Convert to WAV
                '-y',
                str(output_path)
            ]
            
            # Run FFmpeg
            process = subprocess.run(command, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg audio extraction error: {process.stderr}")
                
            self.logger.info(f"Audio extracted successfully to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Audio extraction failed: {str(e)}")

    def trim_video(self, input_path: Path, output_path: Path, 
                 start_time: float, end_time: float, fade_duration: float = 0.5) -> bool:
        """
        Trim video file with optional fade
        
        Args:
            input_path: Source video file path
            output_path: Output file path
            start_time: Start time in seconds
            end_time: End time in seconds
            fade_duration: Duration of fade in/out in seconds
            
        Returns:
            bool: True if successful
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Trimming video: {input_path} -> {output_path}")
            
            # Calculate segment duration for fade effect
            duration = end_time - start_time
            
            # Build FFmpeg command with fade
            fade_video = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={duration-fade_duration}:d={fade_duration}"
            fade_audio = f"afade=t=in:st=0:d={fade_duration},afade=t=out:st={duration-fade_duration}:d={fade_duration}"
            
            command = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-vf', fade_video,
                '-af', fade_audio,
                '-c:v', 'libx264',  # Use H.264 codec for compatibility
                '-c:a', 'aac',  # Use AAC for audio
                '-b:a', '128k',
                '-y',
                str(output_path)
            ]
            
            # Run FFmpeg
            process = subprocess.run(command, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg video trimming error: {process.stderr}")
                
            self.logger.info("Video trimming completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Video trimming failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Video trimming failed: {str(e)}")


    
    def trim_multiple_segments(self, input_path: Path, output_path: Path, 
                             segments: List[Tuple[float, float]], 
                             fade_duration: float = 0.5) -> bool:
        """
        Trim multiple segments from video and concatenate them
        
        Args:
            input_path: Source video file path
            output_path: Output file path
            segments: List of (start_time, end_time) tuples in seconds
            fade_duration: Duration of fade in/out in seconds
            
        Returns:
            bool: True if successful
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Trimming multiple video segments from: {input_path}")
            
            if not segments:
                raise MediaConverterError("No segments provided for trimming")
                
            # Create temporary directory for segments
            temp_dir = Path(os.path.dirname(output_path)) / "temp_segments"
            temp_dir.mkdir(exist_ok=True)
            
            # Create a file to list all segments for concatenation
            concat_file = temp_dir / "concat_list.txt"
            
            # Extract each segment
            segment_files = []
            
            for i, (start_time, end_time) in enumerate(segments):
                segment_path = temp_dir / f"segment_{i}.mp4"
                segment_files.append(segment_path)
                
                # Calculate segment duration for fade effect
                duration = end_time - start_time
                
                # Build FFmpeg command for each segment with fades
                fade_video = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={duration-fade_duration}:d={fade_duration}"
                fade_audio = f"afade=t=in:st=0:d={fade_duration},afade=t=out:st={duration-fade_duration}:d={fade_duration}"
                
                command = [
                    'ffmpeg',
                    '-i', str(input_path),
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-vf', fade_video,
                    '-af', fade_audio,
                    '-c:v', 'libx264',  # Use H.264 for compatibility
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-y',
                    str(segment_path)
                ]
                
                # Run FFmpeg
                process = subprocess.run(command, capture_output=True, text=True)
                
                if process.returncode != 0:
                    raise MediaConverterError(f"FFmpeg segment error: {process.stderr}")
            
            # Create concatenation file
            with open(concat_file, 'w') as f:
                for segment in segment_files:
                    f.write(f"file '{segment}'\n")
            
            # Concatenate all segments
            concat_command = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',  # Try to copy codecs when possible
                '-y',
                str(output_path)
            ]
            
            process = subprocess.run(concat_command, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg concatenation error: {process.stderr}")
            
            # Clean up temporary files
            for segment in segment_files:
                if segment.exists():
                    segment.unlink()
            concat_file.unlink()
            temp_dir.rmdir()
                
            self.logger.info("Multiple segment video trimming completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Multiple segment video trimming failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Multiple segment video trimming failed: {str(e)}")

    def optimize_for_ai(self, input_path: Path, output_path: Path, optimization_level: str, progress_callback=None) -> bool:
        """
        Optimize video for AI processing with different compression levels
        
        Args:
            input_path: Source video file path
            output_path: Output file path
            optimization_level: Level of optimization ('low', 'medium', 'high')
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if successful
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Optimizing video for AI analysis: {input_path}")
            
            # Get the video duration for progress calculation
            info = self.get_video_info(input_path)
            total_duration = info['duration']
            
            # Define optimization parameters based on level
            if optimization_level == 'low':
                # Light optimization - maintains good quality
                video_bitrate = '1000k'
                audio_bitrate = '96k'
                resolution = 'scale=640:-1'
                fps = 15
            elif optimization_level == 'medium':
                # Medium optimization - good balance
                video_bitrate = '500k'
                audio_bitrate = '64k'
                resolution = 'scale=480:-1'
                fps = 10
            elif optimization_level == 'high':
                # Heavy optimization - smallest file size
                video_bitrate = '300k'
                audio_bitrate = '48k'
                resolution = 'scale=320:-1'
                fps = 5
            else:
                # Default to medium
                video_bitrate = '500k'
                audio_bitrate = '64k'
                resolution = 'scale=480:-1'
                fps = 10
            
            # Build FFmpeg command with progress monitoring
            command = [
                'ffmpeg',
                '-i', str(input_path),
                '-vf', f"{resolution},fps={fps}",
                '-c:v', 'libx264',  # Use H.264 for compatibility
                '-preset', 'slow',  # Better compression
                '-crf', '28',       # Quality factor (higher = lower quality)
                '-b:v', video_bitrate,
                '-c:a', 'aac',
                '-b:a', audio_bitrate,
                '-movflags', '+faststart',  # Optimize for web streaming
                '-progress', 'pipe:1',  # Output progress to stdout
                '-y',
                str(output_path)
            ]
            
            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Process progress output
            for line in process.stdout:
                if progress_callback and 'out_time_ms=' in line:
                    # Extract time in milliseconds
                    time_str = line.strip().split('=')[1]
                    try:
                        current_time = float(time_str) / 1000000  # Convert microseconds to seconds
                        progress = min(100, (current_time / total_duration) * 100)
                        progress_callback(progress)
                    except (ValueError, ZeroDivisionError):
                        pass
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                raise MediaConverterError(f"FFmpeg optimization error: {stderr}")
                
            self.logger.info(f"Video optimization completed successfully: {optimization_level} level")
            
            # Log compression results
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            compression_ratio = (1 - (compressed_size / original_size)) * 100
            
            self.logger.info(f"Compression results: {original_size} -> {compressed_size} bytes "
                        f"({compression_ratio:.2f}% reduction)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video optimization failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Video optimization failed: {str(e)}")


    def optimize_audio_for_ai(self, input_path: Path, output_path: Path, optimization_level: str) -> bool:
        """
        Optimize audio for AI processing with different compression levels
        
        Args:
            input_path: Source audio file path
            output_path: Output file path
            optimization_level: Level of optimization ('low', 'medium', 'high')
            
        Returns:
            bool: True if successful
            
        Raises:
            MediaConverterError: If operation fails
        """
        try:
            self.logger.info(f"Optimizing audio for AI analysis: {input_path}")
            
            # Define optimization parameters based on level
            if optimization_level == 'low':
                # Light optimization - maintains good quality
                audio_bitrate = '96k'
                sample_rate = '22050'  # Reduced from CD quality but still good
            elif optimization_level == 'medium':
                # Medium optimization - good balance
                audio_bitrate = '64k'
                sample_rate = '16000'  # Common for speech recognition
            elif optimization_level == 'high':
                # Heavy optimization - smallest file size
                audio_bitrate = '32k'
                sample_rate = '8000'  # Minimum for understandable speech
            else:
                # Default to medium
                audio_bitrate = '64k'
                sample_rate = '16000'
            
            # Ensure the output file has the correct extension
            output_path = output_path.with_suffix('.m4a')  # Use M4A instead of MP3
            
            # Build FFmpeg command with corrected parameters
            command = [
                'ffmpeg',
                '-i', str(input_path),
                '-vn',  # No video
                '-ar', sample_rate,
                '-ac', '1',  # Convert to mono
                '-c:a', 'aac',  # Use AAC instead of MP3 for better compatibility
                '-b:a', audio_bitrate,
                '-y',
                str(output_path)
            ]
            
            # Run FFmpeg
            process = subprocess.run(command, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise MediaConverterError(f"FFmpeg audio optimization error: {process.stderr}")
                
            self.logger.info(f"Audio optimization completed successfully: {optimization_level} level")
            
            # Log compression results
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            compression_ratio = (1 - (compressed_size / original_size)) * 100
            
            self.logger.info(f"Compression results: {original_size} -> {compressed_size} bytes "
                        f"({compression_ratio:.2f}% reduction)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio optimization failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Audio optimization failed: {str(e)}")

    def detect_media_type(self, file_path: Path) -> str:
        """
        Detect if file is video, audio, or screen recording based on metadata
        
        Args:
            file_path: Path to media file
            
        Returns:
            str: Media type ('video', 'audio', 'screen_recording')
            
        Raises:
            MediaConverterError: If detection fails
        """
        try:
            self.logger.info(f"Detecting media type for: {file_path}")
            
            # Get file info
            info = self.get_video_info(file_path)
            
            # Check for audio-only file
            if not info.get('video_codec') or info.get('video_codec') == 'none':
                return 'audio'
                
            # Check for screen recording (common characteristics)
            # Screen recordings often have specific resolutions, low frame rates,
            # and particular codecs
            
            # Common screen recording resolutions
            screen_resolutions = [
                (1920, 1080), (1280, 720), (1366, 768),  # Common laptop/desktop
                (2560, 1440), (3840, 2160),              # Higher-end displays
                (750, 1334), (1080, 1920),               # Mobile portrait
                (1334, 750), (1920, 1080)                # Mobile landscape
            ]
            
            # Common screen recording characteristics
            is_exactly_common_resolution = (info['width'], info['height']) in screen_resolutions
            is_low_framerate = info['fps'] <= 30  # Screen recordings often use lower frame rates
            
            # Also check file metadata for common screen recording software
            metadata_str = json.dumps(self.metadata).lower()
            screen_recording_keywords = [
                'screen', 'capture', 'screencast', 'screenrecording', 
                'quicktime', 'obs', 'camtasia', 'bandicam', 'shadowplay',
                'xbox game bar', 'android', 'ios', 'iphone', 'samsung'
            ]
            
            has_screen_recording_metadata = any(keyword in metadata_str for keyword in screen_recording_keywords)
            
            if (is_exactly_common_resolution and is_low_framerate) or has_screen_recording_metadata:
                return 'screen_recording'
                
            # Default to regular video
            return 'video'
            
        except Exception as e:
            self.logger.error(f"Media type detection failed: {str(e)}", exc_info=True)
            # If detection fails, make best guess based on extension
            ext = file_path.suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
                return 'video'
            elif ext in ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg']:
                return 'audio'
            else:
                return 'unknown'

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
                    
                    # Define a progress callback for this file
                    def update_file_progress(progress):
                        # Calculate overall progress: completed files + current file progress
                        overall = ((i - 1) / total * 100) + (progress / total)
                        self.queue.put(('progress', overall))
                    
                    # Process file with current settings
                    self.converter.process_file(file_path, settings, progress_callback=update_file_progress)
                    
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
                # Try to get duration from FFprobe
                command = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(file_path)
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    total_duration += duration
                else:
                    # If can't read duration, estimate 5 minutes
                    total_duration += 300
            except Exception:
                # If can't read duration, estimate 5 minutes
                total_duration += 300
                
        # Estimate processing time (assuming processing is 2x realtime for audio, 4x for video)
        # Check if any files are video
        has_video = any(file_path.suffix.lower() in 
                     ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
                     for file_path in self.batch_settings.keys())
        
        return total_duration * (4 if has_video else 2)

class MediaConverter:
    """Main application class for media conversion and audio processing"""
    def __init__(self):
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.current_media_file: Optional[Path] = None
        self.media_type: str = 'unknown'  # 'audio', 'video', 'screen_recording', 'unknown'
        self.stop_event = Event()
        
        # Initialize components
        self.analyzer = AudioAnalyzer()
        self.trimmer = AudioTrimmer()
        self.video_processor = VideoProcessor()
        self.batch_processor = BatchProcessor(self)
        
        # Format options
        self.format_options = {
            # Audio formats
            'WAV': {
                'extension': 'wav',
                'codec': 'pcm_s16le',
                'bitrates': ['1411k'],
                'sample_rates': ['44100', '48000', '96000'],
                'type': 'audio'
            },
            'MP3': {
                'extension': 'mp3',
                'codec': 'libmp3lame',
                'bitrates': ['320k', '256k', '192k', '128k', '96k', '64k'],
                'sample_rates': ['44100', '48000', '32000', '22050', '16000'],
                'type': 'audio'
            },
            'AAC': {
                'extension': 'm4a',
                'codec': 'aac',
                'bitrates': ['256k', '192k', '128k', '96k', '64k'],
                'sample_rates': ['44100', '48000', '32000'],
                'type': 'audio'
            },
            'FLAC': {
                'extension': 'flac',
                'codec': 'flac',
                'bitrates': ['0'],  # Lossless
                'sample_rates': ['44100', '48000', '96000'],
                'type': 'audio'
            },
            # Video formats
            'MP4 (H.264)': {
                'extension': 'mp4',
                'codec': 'libx264',
                'bitrates': ['5000k', '2500k', '1500k', '1000k', '500k'],
                'sample_rates': ['44100', '48000'],
                'resolutions': ['original', '1920x1080', '1280x720', '854x480', '640x360'],
                'type': 'video'
            },
            'MP4 (H.265)': {
                'extension': 'mp4',
                'codec': 'libx265',
                'bitrates': ['3000k', '1500k', '1000k', '500k', '250k'],
                'sample_rates': ['44100', '48000'],
                'resolutions': ['original', '1920x1080', '1280x720', '854x480', '640x360'],
                'type': 'video'
            },
            'WebM (VP9)': {
                'extension': 'webm',
                'codec': 'libvpx-vp9',
                'bitrates': ['2000k', '1000k', '500k', '250k'],
                'sample_rates': ['44100', '48000'],
                'resolutions': ['original', '1920x1080', '1280x720', '854x480', '640x360'],
                'type': 'video'
            },
            'GIF': {
                'extension': 'gif',
                'codec': 'gif',
                'fps': ['10', '15', '24'],
                'type': 'video'
            },
            # AI Optimized formats
            'AI Audio (Low Compression)': {
                'extension': 'm4a',  
                'codec': 'aac',      
                'bitrates': ['96k'],
                'sample_rates': ['22050'],
                'type': 'audio',
                'is_ai_optimized': True
            },
            'AI Audio (Medium Compression)': {
                'extension': 'm4a',
                'codec': 'aac', 
                'bitrates': ['64k'],
                'sample_rates': ['16000'],
                'type': 'audio',
                'is_ai_optimized': True
            },
            'AI Audio (High Compression)': {
                'extension': 'm4a',
                'codec': 'aac',
                'bitrates': ['32k'],
                'sample_rates': ['8000'],
                'type': 'audio',
                'is_ai_optimized': True
            },
            'AI Video (Low Compression)': {
                'extension': 'mp4',
                'codec': 'libx264',
                'bitrates': ['1000k'],
                'sample_rates': ['22050'],
                'resolutions': ['640x360'],
                'fps': '15',
                'type': 'video',
                'is_ai_optimized': True
            },
            'AI Video (Medium Compression)': {
                'extension': 'mp4',
                'codec': 'libx264',
                'bitrates': ['500k'],
                'sample_rates': ['16000'],
                'resolutions': ['480x270'],
                'fps': '10',
                'type': 'video',
                'is_ai_optimized': True
            },
            'AI Video (High Compression)': {
                'extension': 'mp4',
                'codec': 'libx264',
                'bitrates': ['300k'],
                'sample_rates': ['8000'],
                'resolutions': ['320x180'],
                'fps': '5',
                'type': 'video',
                'is_ai_optimized': True
            }
        }
        
        # Default values
        self.current_format = 'WAV'
        self.current_bitrate = '320k'
        self.current_sample_rate = '44100'
        self.current_resolution = 'original'
        self.current_fps = '30'
        
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
        self.root.geometry("900x950")  # Slightly larger to accommodate new features
        
        # Create main container with notebook for tabbed interface
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.conversion_tab = ttk.Frame(self.notebook, padding="10")
        self.trimming_tab = ttk.Frame(self.notebook, padding="10")
        self.analysis_tab = ttk.Frame(self.notebook, padding="10")
        self.batch_tab = ttk.Frame(self.notebook, padding="10")
        
        # Add tabs to notebook
        self.notebook.add(self.conversion_tab, text="Conversion")
        self.notebook.add(self.trimming_tab, text="Trimming")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.batch_tab, text="Batch Processing")
        
        # Setup UI sections on each tab
        self.setup_format_section(self.conversion_tab)
        self.setup_path_section(self.conversion_tab)
        self.setup_ai_optimization_section(self.conversion_tab)
        self.setup_progress_section(self.conversion_tab)
        
        self.setup_trim_section(self.trimming_tab)
        self.setup_multi_trim_section(self.trimming_tab)
        
        self.setup_analysis_section(self.analysis_tab)
        
        self.setup_batch_section(self.batch_tab)
        
        # Setup log section (common to all tabs)
        self.setup_log_section(main_container)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(0, weight=1)
        
        # Initialize queue for thread communication
        self.queue = Queue()

    def setup_format_section(self, parent: ttk.Frame) -> None:
        """Setup format selection section"""
        format_frame = ttk.LabelFrame(parent, text="Output Format", padding="5")
        format_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Media type selection
        ttk.Label(format_frame, text="Media Type:").grid(row=0, column=0, sticky=tk.W)
        self.media_type_var = tk.StringVar(value="Auto Detect")
        media_type_combo = ttk.Combobox(
            format_frame, 
            textvariable=self.media_type_var,
            values=["Auto Detect", "Audio", "Video", "Screen Recording"]
        )
        media_type_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        media_type_combo.bind('<<ComboboxSelected>>', self.update_format_options_for_media_type)
        
        # Format selection
        ttk.Label(format_frame, text="Format:").grid(row=1, column=0, sticky=tk.W)
        self.format_var = tk.StringVar(value=self.current_format)
        self.format_combo = ttk.Combobox(
            format_frame, 
            textvariable=self.format_var,
            values=[]  # Will be populated based on media type
        )
        self.format_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.format_combo.bind('<<ComboboxSelected>>', self.update_format_options)
        
        # Bitrate selection
        ttk.Label(format_frame, text="Bitrate:").grid(row=2, column=0, sticky=tk.W)
        self.bitrate_var = tk.StringVar(value=self.current_bitrate)
        self.bitrate_combo = ttk.Combobox(format_frame, textvariable=self.bitrate_var)
        self.bitrate_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Sample rate selection (for audio)
        ttk.Label(format_frame, text="Sample Rate:").grid(row=3, column=0, sticky=tk.W)
        self.sample_rate_var = tk.StringVar(value=self.current_sample_rate)
        self.sample_rate_combo = ttk.Combobox(
            format_frame, 
            textvariable=self.sample_rate_var
        )
        self.sample_rate_combo.grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Resolution selection (for video)
        ttk.Label(format_frame, text="Resolution:").grid(row=4, column=0, sticky=tk.W)
        self.resolution_var = tk.StringVar(value=self.current_resolution)
        self.resolution_combo = ttk.Combobox(
            format_frame,
            textvariable=self.resolution_var
        )
        self.resolution_combo.grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # FPS selection (for video)
        ttk.Label(format_frame, text="Frame Rate:").grid(row=5, column=0, sticky=tk.W)
        self.fps_var = tk.StringVar(value=self.current_fps)
        self.fps_combo = ttk.Combobox(
            format_frame,
            textvariable=self.fps_var,
            values=["original", "60", "30", "24", "15", "10", "5"]
        )
        self.fps_combo.grid(row=5, column=1, sticky=tk.W, padx=5)
        
        # Channel configuration
        ttk.Label(format_frame, text="Audio Channels:").grid(row=6, column=0, sticky=tk.W)
        self.channel_var = tk.StringVar(value="2")
        channel_combo = ttk.Combobox(
            format_frame,
            textvariable=self.channel_var,
            values=["1", "2"]
        )
        channel_combo.grid(row=6, column=1, sticky=tk.W, padx=5)
        
        # Update format options initially
        self.update_format_options_for_media_type()

    def setup_ai_optimization_section(self, parent: ttk.Frame) -> None:
        """Setup AI optimization options"""
        ai_frame = ttk.LabelFrame(parent, text="AI Optimization", padding="5")
        ai_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Enable AI optimization checkbox
        self.ai_optimize_var = tk.BooleanVar(value=False)
        ai_optimize_check = ttk.Checkbutton(
            ai_frame,
            text="Optimize for AI Analysis",
            variable=self.ai_optimize_var,
            command=self.toggle_ai_optimization
        )
        ai_optimize_check.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Optimization level
        ttk.Label(ai_frame, text="Compression Level:").grid(row=1, column=0, sticky=tk.W)
        self.ai_level_var = tk.StringVar(value="medium")
        ai_level_frame = ttk.Frame(ai_frame)
        ai_level_frame.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Radiobutton(
            ai_level_frame, 
            text="Low", 
            variable=self.ai_level_var,
            value="low"
        ).grid(row=0, column=0, padx=5)
        
        ttk.Radiobutton(
            ai_level_frame, 
            text="Medium", 
            variable=self.ai_level_var,
            value="medium"
        ).grid(row=0, column=1, padx=5)
        
        ttk.Radiobutton(
            ai_level_frame, 
            text="High", 
            variable=self.ai_level_var,
            value="high"
        ).grid(row=0, column=2, padx=5)
        
        # Optimization description
        optimization_desc = (
            "Low: Better quality, larger file size\n"
            "Medium: Balanced quality and size\n"
            "High: Smaller file size, reduced quality"
        )
        self.ai_desc_var = tk.StringVar(value=optimization_desc)
        ttk.Label(ai_frame, textvariable=self.ai_desc_var, justify=tk.LEFT).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=5
        )

    def setup_path_section(self, parent: ttk.Frame) -> None:
        """Setup file path selection section"""
        path_frame = ttk.LabelFrame(parent, text="File Selection", padding="5")
        path_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
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
        
        # Media info display
        self.media_info_var = tk.StringVar(value="No file selected")
        ttk.Label(path_frame, textvariable=self.media_info_var, justify=tk.LEFT).grid(
            row=2, column=0, sticky=tk.W, pady=5
        )

    def setup_trim_section(self, parent: ttk.Frame) -> None:
        """Setup audio trimming section"""
        trim_frame = ttk.LabelFrame(parent, text="Trim Media", padding="5")
        trim_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
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

    def setup_multi_trim_section(self, parent: ttk.Frame) -> None:
        """Setup multi-segment trimming section"""
        multi_trim_frame = ttk.LabelFrame(parent, text="Multi-Segment Trimming", padding="5")
        multi_trim_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Multiple selections list
        ttk.Label(multi_trim_frame, text="Selected Segments:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        segments_frame = ttk.Frame(multi_trim_frame)
        segments_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Listbox to display segments
        self.segments_list = tk.Listbox(segments_frame, height=5, width=50)
        self.segments_list.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for segments list
        segments_scroll = ttk.Scrollbar(segments_frame, orient="vertical",
                                     command=self.segments_list.yview)
        segments_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.segments_list.configure(yscrollcommand=segments_scroll.set)
        
        # Buttons for managing segments
        button_frame = ttk.Frame(multi_trim_frame)
        button_frame.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        ttk.Button(button_frame, text="Add Current Selection",
                 command=self.add_segment).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Remove Selected",
                 command=self.remove_segment).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Clear All",
                 command=self.clear_segments).grid(row=0, column=2, padx=5)
        
        # Process segments
        process_frame = ttk.Frame(multi_trim_frame)
        process_frame.grid(row=3, column=0, sticky=tk.W, pady=5)
        
        ttk.Button(process_frame, text="Combine Selected Segments",
                 command=self.process_segments).grid(row=0, column=0, padx=5)
        
        # Fade options
        fade_frame = ttk.Frame(multi_trim_frame)
        fade_frame.grid(row=4, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(fade_frame, text="Fade Duration (sec):").grid(row=0, column=0, sticky=tk.W)
        self.fade_duration_var = tk.StringVar(value="0.5")
        ttk.Entry(fade_frame, textvariable=self.fade_duration_var, width=5).grid(row=0, column=1, padx=5)
        
        # Crossfade between segments
        self.crossfade_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fade_frame, text="Apply crossfade between segments", 
                      variable=self.crossfade_var).grid(row=0, column=2, padx=5)

    def setup_analysis_section(self, parent: ttk.Frame) -> None:
        """Setup audio analysis section"""
        analysis_frame = ttk.LabelFrame(parent, text="Media Analysis", padding="5")
        analysis_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Analysis controls
        control_frame = ttk.Frame(analysis_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(control_frame, text="Analyze Media",
                  command=self.analyze_current_file).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Export Analysis",
                  command=self.export_analysis).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Detect Silence",
                  command=self.detect_silence).grid(row=0, column=2, padx=5)
        
        # Additional controls for video
        video_controls = ttk.Frame(analysis_frame)
        video_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(video_controls, text="Extract Audio Track",
                 command=self.extract_audio_track).grid(row=0, column=0, padx=5)
        ttk.Button(video_controls, text="Extract Frame at Current Time",
                 command=self.extract_current_frame).grid(row=0, column=1, padx=5)
        
        # Analysis display
        self.analysis_text = tk.Text(analysis_frame, height=12, width=70)
        self.analysis_text.grid(row=2, column=0, pady=5)
        
        # Scrollbar for analysis text
        analysis_scroll = ttk.Scrollbar(analysis_frame, orient="vertical",
                                      command=self.analysis_text.yview)
        analysis_scroll.grid(row=2, column=1, sticky=(tk.N, tk.S))
        self.analysis_text.configure(yscrollcommand=analysis_scroll.set)

    def setup_batch_section(self, parent: ttk.Frame) -> None:
        """Setup batch processing section"""
        batch_frame = ttk.LabelFrame(parent, text="Batch Processing", padding="5")
        batch_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Batch controls
        control_frame = ttk.Frame(batch_frame)
        control_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(control_frame, text="Add to Batch",
                  command=self.add_to_batch).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Add Folder",
                  command=self.add_folder_to_batch).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Remove Selected",
                  command=self.remove_from_batch).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Clear Batch",
                  command=self.clear_batch).grid(row=0, column=3, padx=5)
        
        # Batch queue display
        batch_display_frame = ttk.Frame(batch_frame)
        batch_display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.batch_list = tk.Listbox(batch_display_frame, height=10, width=70)
        self.batch_list.grid(row=0, column=0, pady=5)
        
        # Scrollbar for batch list
        batch_scroll = ttk.Scrollbar(batch_display_frame, orient="vertical",
                                   command=self.batch_list.yview)
        batch_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.batch_list.configure(yscrollcommand=batch_scroll.set)
        
        # Batch processing options
        options_frame = ttk.Frame(batch_frame)
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Apply same settings to all files
        self.same_settings_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Apply same settings to all files", 
                      variable=self.same_settings_var).grid(row=0, column=0, sticky=tk.W)
        
        # Process button
        process_frame = ttk.Frame(batch_frame)
        process_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(process_frame, text="Process Batch",
                 command=self.process_batch).grid(row=0, column=0, padx=5)
        
        # Progress and status for batch
        progress_frame = ttk.Frame(batch_frame)
        progress_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(progress_frame, text="Batch Progress:").grid(row=0, column=0, sticky=tk.W)
        
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=400,
            mode="determinate",
            variable=self.batch_progress_var
        )
        self.batch_progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.batch_status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.batch_status_var).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=5
        )

    def setup_progress_section(self, parent: ttk.Frame) -> None:
        """Setup progress tracking section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
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
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
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

    def toggle_ai_optimization(self) -> None:
        """Toggle AI optimization options"""
        if self.ai_optimize_var.get():
            # Update format combo to show AI-optimized formats
            media_type = self.detect_media_type()
            self.update_format_options_for_media_type(media_type=media_type, ai_optimized=True)
        else:
            # Restore normal formats
            self.update_format_options_for_media_type()

    def update_format_options_for_media_type(self, event=None, media_type=None, ai_optimized=False) -> None:
        """Update available formats based on media type"""
        if media_type is None:
            # Get from UI or detect from file
            if self.media_type_var.get() == "Auto Detect":
                media_type = self.detect_media_type()
            else:
                media_type = self.media_type_var.get().lower()
        
        # Filter formats based on media type and AI optimization
        filtered_formats = []
        for format_name, format_info in self.format_options.items():
            # Check if format is for current media type
            if (format_info.get('type', '').lower() == media_type.lower() or 
                media_type == 'auto detect'):
                
                # Check for AI optimization filter
                if ai_optimized:
                    if format_info.get('is_ai_optimized', False):
                        filtered_formats.append(format_name)
                else:
                    if not format_info.get('is_ai_optimized', False):
                        filtered_formats.append(format_name)
        
        # Update format combo
        self.format_combo['values'] = filtered_formats
        
        # Set a default format if current is not valid
        if not filtered_formats:
            return  # No formats available
            
        if self.format_var.get() not in filtered_formats:
            self.format_var.set(filtered_formats[0])
        
        # Update other options based on selected format
        self.update_format_options()

    def update_format_options(self, event=None) -> None:
        """Update available options based on selected format"""
        format_name = self.format_var.get()
        if not format_name or format_name not in self.format_options:
            return
            
        format_info = self.format_options[format_name]
        
        # Update bitrate options
        if 'bitrates' in format_info:
            self.bitrate_combo['values'] = format_info['bitrates']
            if self.bitrate_var.get() not in format_info['bitrates']:
                self.bitrate_var.set(format_info['bitrates'][0])
        
        # Update sample rate options for audio
        if 'sample_rates' in format_info:
            self.sample_rate_combo['values'] = format_info['sample_rates']
            if self.sample_rate_var.get() not in format_info['sample_rates']:
                self.sample_rate_var.set(format_info['sample_rates'][0])
        
        # Update resolution options for video
        if 'resolutions' in format_info:
            self.resolution_combo['values'] = format_info['resolutions']
            if self.resolution_var.get() not in format_info['resolutions']:
                self.resolution_var.set(format_info['resolutions'][0])
        
        # Update FPS options for video
        if 'fps' in format_info and isinstance(format_info['fps'], list):
            self.fps_combo['values'] = format_info['fps']
            if self.fps_var.get() not in format_info['fps']:
                self.fps_var.set(format_info['fps'][0])

    def detect_media_type(self) -> str:
        """Detect if current file is audio, video, or screen recording"""
        if not self.current_media_file:
            return "audio"  # Default to audio if no file
        
        try:
            # Get file extension
            ext = self.current_media_file.suffix.lower()
            
            # Quick check based on extension
            if ext in ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg']:
                self.media_type = 'audio'
                return 'audio'
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
                # Use more detailed detection for video files
                media_type = self.video_processor.detect_media_type(self.current_media_file)
                self.media_type = media_type
                return media_type
            else:
                # Try to get more information using FFprobe
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 
                     'stream=codec_type', '-of', 'csv=p=0', str(self.current_media_file)],
                    capture_output=True, text=True
                )
                
                if 'video' in result.stdout:
                    self.media_type = 'video'
                    return 'video'
                elif 'audio' in result.stdout:
                    self.media_type = 'audio'
                    return 'audio'
                
            # Default case
            self.media_type = 'audio'
            return 'audio'
            
        except Exception as e:
            self.logger.warning(f"Media type detection failed: {str(e)}")
            self.media_type = 'audio'  # Default to audio
            return 'audio'

    def toggle_trim_mode(self) -> None:
        """Switch between simple and advanced trimming modes"""
        if self.trim_mode.get() == "simple":
            self.advanced_trim_frame.grid_remove()
            self.simple_trim_frame.grid()
        else:
            self.simple_trim_frame.grid_remove()
            self.advanced_trim_frame.grid()
            if self.current_media_file:
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

    def add_segment(self) -> None:
        """Add current selection as a segment for multi-trim"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            start = self.parse_time(self.start_time_var.get())
            end = self.parse_time(self.end_time_var.get())
            
            if start >= end:
                raise MediaConverterError("Start time must be less than end time")
                
            # Add to trimmer and visualizer
            self.trimmer.add_selection(start, end)
            if self.trim_mode.get() == "advanced":
                self.waveform_visualizer.add_selection(start, end)
            
            # Add to display list
            segment_text = f"{self.format_time(start)} to {self.format_time(end)}"
            self.segments_list.insert(tk.END, segment_text)
            
            self.log_message(f"Added segment: {segment_text}")
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def remove_segment(self) -> None:
        """Remove selected segment from multi-trim list"""
        selection = self.segments_list.curselection()
        if selection:
            index = selection[0]
            # Remove from listbox
            self.segments_list.delete(index)
            
            # Update the trimmer selections (rebuild list from UI)
            self.trimmer.clear_selections()
            for i in range(self.segments_list.size()):
                segment_text = self.segments_list.get(i)
                start_str, end_str = segment_text.split(" to ")
                start_time = self.parse_time(start_str)
                end_time = self.parse_time(end_str)
                self.trimmer.add_selection(start_time, end_time)
            
            # Clear and redraw visualizer if in advanced mode
            if self.trim_mode.get() == "advanced":
                self.waveform_visualizer.clear_selections()
                for start_time, end_time in self.trimmer.selections:
                    self.waveform_visualizer.add_selection(start_time, end_time)
            
            self.log_message(f"Removed segment at index {index}")


    def clear_segments(self) -> None:
        """Clear all segments from multi-trim list"""
        self.segments_list.delete(0, tk.END)
        self.trimmer.clear_selections()
        
        # Clear visualizer selections if in advanced mode
        if self.trim_mode.get() == "advanced":
            self.waveform_visualizer.clear_selections()
            self.load_waveform()  # Reload the waveform
        
        self.log_message("Cleared all segments")

    def process_segments(self) -> None:
        """Process multiple segments into a single output file"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            if not self.trimmer.selections:
                raise MediaConverterError("No segments have been selected")
                
            if not self.output_path:
                raise MediaConverterError("No output path selected")
                
            # Get output filename
            output_filename = self.current_media_file.stem + "_combined"
            output_ext = self.format_options[self.format_var.get()]['extension']
            output_file = self.output_path / f"{output_filename}.{output_ext}"
            
            # Get fade duration
            try:
                fade_duration = float(self.fade_duration_var.get())
                if fade_duration < 0:
                    fade_duration = 0
            except ValueError:
                fade_duration = 0.5
            
            self.status_var.set("Processing segments...")
            self.progress_var.set(10)
            
            # Process based on media type
            if self.media_type in ['video', 'screen_recording']:
                success = self.video_processor.trim_multiple_segments(
                    self.current_media_file,
                    output_file,
                    self.trimmer.selections,
                    fade_duration
                )
            else:  # Audio
                success = self.trimmer.trim_multiple_segments(
                    self.current_media_file,
                    output_file,
                    self.trimmer.selections,
                    fade_duration
                )
            
            if success:
                self.status_var.set("Segments processed successfully")
                self.progress_var.set(100)
                self.log_message(f"Created combined file: {output_file}")
                messagebox.showinfo("Success", f"Segments combined into:\n{output_file}")
            else:
                raise MediaConverterError("Failed to process segments")
                
        except MediaConverterError as e:
            self.status_var.set("Error processing segments")
            messagebox.showerror("Error", str(e))
            self.log_message(f"Error processing segments: {str(e)}")

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
            if self.media_type in ['video', 'screen_recording']:
                # For video, extract the audio first
                temp_audio_file = self.current_media_file.parent / f"temp_audio_{self.current_media_file.stem}.wav"
                if self.video_processor.extract_audio(self.current_media_file, temp_audio_file):
                    # Load the audio data
                    self.trimmer.load_audio_file(temp_audio_file)
                    self.waveform_visualizer.plot_waveform(
                        self.trimmer.waveform_data,
                        self.trimmer.sample_rate
                    )
                    self.timeline.configure(to=self.trimmer.duration)
                    
                    # Store for future reference
                    self.waveform_visualizer.last_waveform_data = self.trimmer.waveform_data
                    self.waveform_visualizer.last_sample_rate = self.trimmer.sample_rate
                    
                    # Remove temp file
                    if temp_audio_file.exists():
                        temp_audio_file.unlink()
            else:
                # For audio, load directly
                if self.trimmer.load_audio_file(self.current_media_file):
                    self.waveform_visualizer.plot_waveform(
                        self.trimmer.waveform_data,
                        self.trimmer.sample_rate
                    )
                    self.timeline.configure(to=self.trimmer.duration)
                    
                    # Store for future reference
                    self.waveform_visualizer.last_waveform_data = self.trimmer.waveform_data
                    self.waveform_visualizer.last_sample_rate = self.trimmer.sample_rate
                    
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def get_file_duration(self, file_path: Path) -> float:
        """Get duration of media file using FFmpeg"""
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
                        ("All media files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wav *.mp3 *.aac *.m4a *.flac *.ogg"),
                        ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv"),
                        ("Audio files", "*.wav *.mp3 *.aac *.m4a *.flac *.ogg"),
                        ("All files", "*.*")
                    ]
                )
            else:
                path = filedialog.askdirectory(title="Select input folder")
                
            if path:
                self.input_path = Path(path)
                self.input_path_var.set(str(self.input_path))
                self.current_media_file = self.input_path if self.input_path.is_file() else None
                self.validate_input_path()
                
                # If a single file, detect media type and update info
                if self.input_path.is_file():
                    # Detect media type
                    self.media_type = self.detect_media_type()
                    self.update_format_options_for_media_type()
                    
                    # Get media info
                    if self.media_type in ['video', 'screen_recording']:
                        media_info = self.video_processor.get_video_info(self.input_path)
                        duration = media_info['duration']
                        info_text = (
                            f"File: {self.input_path.name}\n"
                            f"Type: {self.media_type.capitalize()}\n"
                            f"Duration: {self.format_time(duration)}\n"
                            f"Resolution: {media_info['width']}x{media_info['height']}\n"
                            f"FPS: {media_info['fps']:.2f}\n"
                            f"Video Codec: {media_info['video_codec']}\n"
                            f"Audio Codec: {media_info['audio_codec']}"
                        )
                    else:  # Audio
                        duration = self.get_file_duration(self.input_path)
                        info_text = (
                            f"File: {self.input_path.name}\n"
                            f"Type: Audio\n"
                            f"Duration: {self.format_time(duration)}"
                        )
                    
                    # Update UI with media info
                    self.media_info_var.set(info_text)
                    
                    # Set end time for trimming
                    if duration > 0:
                        self.end_time_var.set(self.format_time(duration))
                
                self.log_message(f"Selected input: {self.input_path}")
                
                # Load waveform if in advanced trim mode
                if self.trim_mode.get() == "advanced" and self.current_media_file:
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

    def extract_audio_track(self) -> None:
        """Extract audio track from current video file"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            if self.media_type not in ['video', 'screen_recording']:
                raise MediaConverterError("Current file is not a video")
                
            if not self.output_path:
                raise MediaConverterError("No output path selected")
                
            # Create output path
            output_file = self.output_path / f"{self.current_media_file.stem}_audio.wav"
            
            self.status_var.set("Extracting audio...")
            
            if self.video_processor.extract_audio(self.current_media_file, output_file):
                self.status_var.set("Audio extracted successfully")
                self.log_message(f"Audio extracted to: {output_file}")
                messagebox.showinfo("Success", f"Audio extracted to:\n{output_file}")
            else:
                raise MediaConverterError("Failed to extract audio")
                
        except MediaConverterError as e:
            self.status_var.set("Error extracting audio")
            messagebox.showerror("Error", str(e))
            self.log_message(f"Error extracting audio: {str(e)}")

    def extract_current_frame(self) -> None:
        """Extract frame at current timeline position from video"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            if self.media_type not in ['video', 'screen_recording']:
                raise MediaConverterError("Current file is not a video")
                
            if not self.output_path:
                raise MediaConverterError("No output path selected")
                
            # Get current time position
            current_time = float(self.timeline_var.get()) / 100 * self.trimmer.duration
            
            # Create output path
            output_file = self.output_path / f"{self.current_media_file.stem}_frame_{int(current_time)}.jpg"
            
            self.status_var.set("Extracting frame...")
            
            if self.video_processor.extract_frame(self.current_media_file, current_time, output_file):
                self.status_var.set("Frame extracted successfully")
                self.log_message(f"Frame extracted to: {output_file}")
                messagebox.showinfo("Success", f"Frame extracted to:\n{output_file}")
            else:
                raise MediaConverterError("Failed to extract frame")
                
        except MediaConverterError as e:
            self.status_var.set("Error extracting frame")
            messagebox.showerror("Error", str(e))
            self.log_message(f"Error extracting frame: {str(e)}")

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
            '.wav', '.mp3', '.aac', '.m4a', '.flac', '.ogg',
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'
        )

    def get_media_files(self, directory: Path) -> Iterator[Path]:
        """Generator for supported media files in directory"""
        for path in directory.glob('*'):
            if self.is_media_file(path):
                yield path

    def apply_trim(self) -> None:
        """Apply trim settings to the current media file"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            start = self.parse_time(self.start_time_var.get())
            end = self.parse_time(self.end_time_var.get())
            
            if start >= end:
                raise MediaConverterError("Start time must be less than end time")
                
            if not self.output_path:
                raise MediaConverterError("No output path selected")
                
            output_file = self.output_path / f"trimmed_{self.current_media_file.name}"
            
            # Get fade duration
            try:
                fade_duration = float(self.fade_duration_var.get())
            except ValueError:
                fade_duration = 0.5
                
            self.status_var.set("Trimming media...")
            
            # Use appropriate trimmer based on media type
            if self.media_type in ['video', 'screen_recording']:
                success = self.video_processor.trim_video(
                    self.current_media_file, output_file, start, end, fade_duration
                )
            else:  # Audio
                success = self.trimmer.trim_audio(
                    self.current_media_file, output_file, start, end, fade_duration
                )
                
            if success:
                self.status_var.set("Trim completed successfully")
                self.log_message(f"Successfully trimmed: {output_file}")
                messagebox.showinfo("Success", "Trim operation completed successfully")
            else:
                raise MediaConverterError("Trim operation failed")
                
        except MediaConverterError as e:
            self.status_var.set("Error trimming media")
            messagebox.showerror("Error", str(e))
            self.log_message(f"Error trimming media: {str(e)}")

    def apply_preset(self, event=None) -> None:
        """Apply selected trimming preset"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
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
                duration = self.get_file_duration(self.current_media_file)
                if duration > 30:
                    start = duration - 30
                    self.start_time_var.set(self.format_time(start))
                    self.end_time_var.set(self.format_time(duration))
            
            # Update waveform selection if in advanced mode
            if self.trim_mode.get() == "advanced":
                start = self.parse_time(self.start_time_var.get())
                end = self.parse_time(self.end_time_var.get())
                self.waveform_visualizer.update_selection(start, end)
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

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
                    # Define a progress callback for this file
                    def update_file_progress(progress):
                        # Calculate overall progress: completed files + current file progress
                        overall = ((i - 1) / total_files * 100) + (progress / total_files)
                        self.queue.put(('progress', overall))
                    
                    self.process_file(file_path, progress_callback=update_file_progress)
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


    # Add a progress callback parameter to process_file
    def process_file(self, file_path: Path, settings: Optional[Dict] = None, progress_callback=None) -> None:
        """
        Process a single file with conversion and trimming
        Args:
            file_path: Path to input file
            settings: Optional dictionary of processing settings
            progress_callback: Optional callback function for progress updates
        """
        try:
            # Determine media type
            media_type = self.video_processor.detect_media_type(file_path)
            
            # Use provided settings or current UI settings
            if settings is None:
                settings = {
                    'start_time': self.start_time_var.get(),
                    'end_time': self.end_time_var.get(),
                    'format': self.format_var.get(),
                    'bitrate': self.bitrate_var.get(),
                    'sample_rate': self.sample_rate_var.get(),
                    'channels': self.channel_var.get(),
                    'ai_optimize': self.ai_optimize_var.get(),
                    'ai_level': self.ai_level_var.get()
                }
                
                # Add video settings if applicable
                if media_type in ['video', 'screen_recording']:
                    settings.update({
                        'resolution': self.resolution_var.get(),
                        'fps': self.fps_var.get()
                    })
                    
            # Get format info
            format_info = self.format_options[settings['format']]
            output_file = self.output_path / f"{file_path.stem}.{format_info['extension']}"

            # AI optimization takes precedence if enabled
            if settings.get('ai_optimize', False):
                if media_type in ['video', 'screen_recording']:
                    # Ensure proper extension
                    output_file = output_file.with_suffix('.mp4')
                    return self.video_processor.optimize_for_ai(
                        file_path, output_file, settings['ai_level'], progress_callback
                    )
                else:  # Audio
                    # Ensure proper extension
                    output_file = output_file.with_suffix('.m4a')
                    return self.video_processor.optimize_audio_for_ai(
                        file_path, output_file, settings['ai_level'], progress_callback
                    )
            
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

            # Add video parameters if this is a video file
            if media_type in ['video', 'screen_recording'] and format_info.get('type') == 'video':
                # Video codec
                command.extend(['-c:v', format_info['codec']])
                
                # Video bitrate
                command.extend(['-b:v', settings['bitrate']])
                
                # Resolution (if not original)
                if settings.get('resolution') and settings['resolution'] != 'original':
                    command.extend(['-vf', f'scale={settings["resolution"].replace("x", ":")}'])
                
                # FPS (if not original)
                if settings.get('fps') and settings['fps'] != 'original':
                    command.extend(['-r', settings['fps']])
                
                # Add audio codec for video files
                command.extend([
                    '-c:a', 'aac',
                    '-b:a', '128k'
                ])
            else:
                # Audio-only parameters
                command.extend([
                    '-c:a', format_info['codec'],
                    '-ar', settings['sample_rate'],
                    '-ac', settings['channels'],
                    '-b:a', settings['bitrate']
                ])

            # Add progress monitoring
            command.extend([
                '-progress', 'pipe:1',  # Output progress to stdout
                '-y',  # Overwrite output
                str(output_file)
            ])

            # Log the command for debugging
            self.logger.info(f"FFmpeg command: {' '.join(command)}")

            # Run conversion with progress monitoring
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Process progress output
            total_duration = file_duration
            if total_duration <= 0:
                total_duration = 100  # Default if we can't get duration
                
            for line in process.stdout:
                if progress_callback and 'out_time_ms=' in line:
                    # Extract time in milliseconds
                    time_str = line.strip().split('=')[1]
                    try:
                        current_time = float(time_str) / 1000000  # Convert microseconds to seconds
                        progress = min(100, (current_time / total_duration) * 100)
                        progress_callback(progress)
                    except (ValueError, ZeroDivisionError):
                        pass
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                raise MediaConverterError(f"FFmpeg error: {stderr}")
                
            return True

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise MediaConverterError(f"Processing failed: {str(e)}")


    def preview_trim(self) -> None:
        """Preview the trimmed media"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            start = self.parse_time(self.start_time_var.get())
            end = self.parse_time(self.end_time_var.get())
            
            # Create temporary file for preview
            temp_dir = self.current_media_file.parent / "temp_preview"
            temp_dir.mkdir(exist_ok=True)
            
            media_type = self.media_type
            ext = '.mp4' if media_type in ['video', 'screen_recording'] else '.wav'
            temp_output = temp_dir / f"preview_{self.current_media_file.stem}{ext}"
            
            # Use appropriate trimmer based on media type
            if media_type in ['video', 'screen_recording']:
                success = self.video_processor.trim_video(
                    self.current_media_file, temp_output, start, end, 0.3
                )
            else:  # Audio
                success = self.trimmer.trim_audio(
                    self.current_media_file, temp_output, start, end, 0.1
                )
                
            if success:
                # Play preview using ffplay
                subprocess.Popen(['ffplay', '-autoexit', str(temp_output)])
                self.log_message("Playing preview...")
                
                # Schedule cleanup of temp files
                def cleanup():
                    try:
                        if temp_output.exists():
                            temp_output.unlink()
                        if temp_dir.exists():
                            temp_dir.rmdir()
                    except Exception:
                        pass
                
                self.root.after(10000, cleanup)  # Cleanup after 10 seconds
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def add_to_batch(self) -> None:
        """Add current file and settings to batch"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            # Get current settings
            media_type = self.media_type
            
            settings = {
                'start_time': self.start_time_var.get(),
                'end_time': self.end_time_var.get(),
                'format': self.format_var.get(),
                'bitrate': self.bitrate_var.get(),
                'sample_rate': self.sample_rate_var.get(),
                'channels': self.channel_var.get(),
                'ai_optimize': self.ai_optimize_var.get(),
                'ai_level': self.ai_level_var.get()
            }
            
            # Add video settings if applicable
            if media_type in ['video', 'screen_recording']:
                settings.update({
                    'resolution': self.resolution_var.get(),
                    'fps': self.fps_var.get()
                })
            
            self.batch_processor.add_to_batch(self.current_media_file, settings)
            self.batch_list.insert(tk.END, f"{self.current_media_file.name} [{self.format_var.get()}]")
            self.log_message(f"Added to batch: {self.current_media_file.name}")
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

    def add_folder_to_batch(self) -> None:
        """Add all media files from a folder to batch"""
        try:
            folder_path = filedialog.askdirectory(title="Select folder with media files")
            if not folder_path:
                return  # User cancelled
                
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                raise MediaConverterError("Invalid folder selected")
                
            # Get all media files in folder
            media_files = list(self.get_media_files(folder))
            
            if not media_files:
                raise MediaConverterError("No supported media files found in the selected folder")
                
            # Get current settings to apply to all files
            settings = {
                'start_time': "00:00:00",  # No trimming by default for batch folder
                'end_time': "23:59:59",
                'format': self.format_var.get(),
                'bitrate': self.bitrate_var.get(),
                'sample_rate': self.sample_rate_var.get(),
                'channels': self.channel_var.get(),
                'resolution': self.resolution_var.get(),
                'fps': self.fps_var.get(),
                'ai_optimize': self.ai_optimize_var.get(),
                'ai_level': self.ai_level_var.get()
            }
            
            # Add all files to batch
            for file_path in media_files:
                self.batch_processor.add_to_batch(file_path, settings)
                self.batch_list.insert(tk.END, f"{file_path.name} [{self.format_var.get()}]")
                
            self.log_message(f"Added {len(media_files)} files from folder: {folder}")
            messagebox.showinfo("Batch Processing", f"Added {len(media_files)} files to batch queue")
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))


    def remove_from_batch(self) -> None:
        """Remove selected file from batch queue"""
        selection = self.batch_list.curselection()
        if selection:
            index = selection[0]
            filename = self.batch_list.get(index).split(" [")[0]  # Remove format suffix
            
            # Find the file in batch processor
            for file_path in self.batch_processor.get_batch_files():
                if file_path.name == filename:
                    self.batch_processor.remove_from_batch(file_path)
                    self.batch_list.delete(index)
                    self.log_message(f"Removed from batch: {filename}")
                    break

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
                    self.batch_progress_var.set(data)
                elif msg_type == 'status':
                    self.batch_status_var.set(data)
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

    def cancel_conversion(self) -> None:
        """Cancel the ongoing conversion process"""
        self.stop_event.set()
        self.batch_processor.stop_processing()
        self.cancel_btn.state(['disabled'])
        self.status_var.set("Cancelling...")
        self.log_message("Conversion cancelled by user")

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
        """Analyze current media file"""
        try:
            if not self.current_media_file:
                raise MediaConverterError("No media file loaded")
                
            self.status_var.set("Analyzing media...")
            
            # Different analysis based on media type
            if self.media_type in ['video', 'screen_recording']:
                # For video, get detailed info and extract audio for analysis
                video_info = self.video_processor.get_video_info(self.current_media_file)
                
                # Create a temporary file for audio extraction
                temp_audio = self.current_media_file.parent / f"temp_audio_{self.current_media_file.stem}.wav"
                
                # Extract audio for analysis
                if self.video_processor.extract_audio(self.current_media_file, temp_audio):
                    # Analyze the extracted audio
                    audio_analysis_success = self.analyzer.analyze_audio(temp_audio)
                    
                    # Remove temporary file
                    if temp_audio.exists():
                        temp_audio.unlink()
                        
                    if audio_analysis_success:
                        # Combine video info and audio analysis
                        self.display_video_analysis(video_info)
                    else:
                        raise MediaConverterError("Failed to analyze audio track")
                else:
                    raise MediaConverterError("Failed to extract audio for analysis")
            else:
                # For audio files, use regular audio analyzer
                if self.analyzer.analyze_audio(self.current_media_file):
                    self.display_audio_analysis()
                else:
                    raise MediaConverterError("Failed to analyze audio file")
                    
            self.status_var.set("Analysis complete")
                
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Analysis failed")

    def display_audio_analysis(self) -> None:
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
Perceived Loudness: {info['audio_features']['perceived_loudness']:.2f} dB

Silent Intervals: {len(info.get('silent_intervals', []))}
        """
        
        self.analysis_text.insert(tk.END, display_text)

    def display_video_analysis(self, video_info: Dict) -> None:
        """Display video analysis results with audio analysis"""
        self.analysis_text.delete(1.0, tk.END)
        
        # Get the audio analysis info
        audio_info = self.analyzer.audio_info
        
        # Display combined video and audio analysis
        display_text = f"""
Video Analysis Results:
----------------------
Duration: {video_info['duration']:.2f} seconds
Resolution: {video_info['width']}x{video_info['height']}
Frame Rate: {video_info['fps']:.2f} FPS
Bitrate: {int(video_info['bitrate']) // 1000} kbps
Video Codec: {video_info['video_codec']}
Audio Codec: {video_info['audio_codec']}
File Size: {video_info['file_size'] // (1024*1024):.1f} MB

Audio Track Analysis:
-------------------
Sample Rate: {audio_info['sample_rate']} Hz
Tempo: {audio_info['tempo']:.2f} BPM
Average Volume: {audio_info['audio_features']['avg_volume']:.2f}
Perceived Loudness: {audio_info['audio_features']['perceived_loudness']:.2f} dB

Silent Intervals: {len(audio_info.get('silent_intervals', []))}
        """
        
        self.analysis_text.insert(tk.END, display_text)

    def export_analysis(self) -> None:
        """Export audio analysis to JSON"""
        try:
            if not hasattr(self, 'analyzer') or not self.analyzer.audio_info:
                raise MediaConverterError("No analysis data available")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                initialfile=f"media_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if file_path:
                self.analyzer.export_analysis(Path(file_path))
                self.log_message(f"Analysis exported to: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def detect_silence(self) -> None:
        """Detect and mark silent intervals"""
        try:
            if not hasattr(self, 'analyzer') or not self.analyzer.audio_info:
                self.analyze_current_file()
                
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
            
            # Ask if user wants to add silent regions to segment list
            if len(silence_regions) > 0:
                add_regions = messagebox.askyesno(
                    "Silent Regions",
                    "Would you like to add the non-silent regions as segments for trimming?"
                )
                
                if add_regions:
                    # Clear existing segments
                    self.clear_segments()
                    
                    # Add non-silent regions as segments
                    duration = self.analyzer.audio_info['duration']
                    prev_end = 0
                    
                    for region in silence_regions:
                        # Add segment from previous end to current start if gap exists
                        if region['start'] > prev_end:
                            self.trimmer.add_selection(prev_end, region['start'])
                            
                            # Add to segments list
                            segment_text = f"{self.format_time(prev_end)} to {self.format_time(region['start'])}"
                            self.segments_list.insert(tk.END, segment_text)
                            
                        prev_end = region['end']
                    
                    # Add final segment if needed
                    if prev_end < duration:
                        self.trimmer.add_selection(prev_end, duration)
                        
                        # Add to segments list
                        segment_text = f"{self.format_time(prev_end)} to {self.format_time(duration)}"
                        self.segments_list.insert(tk.END, segment_text)
                    
                    # Update waveform if in advanced mode
                    if self.trim_mode.get() == "advanced":
                        self.waveform_visualizer.clear_selections()
                        for start_time, end_time in self.trimmer.selections:
                            self.waveform_visualizer.add_selection(start_time, end_time)
                    
                    self.log_message(f"Added {self.segments_list.size()} non-silent segments")
            
        except MediaConverterError as e:
            messagebox.showerror("Error", str(e))

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