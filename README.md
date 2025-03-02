# Advanced Media Converter

An advanced audio and video processing application built with Python that provides comprehensive media file manipulation, analysis, and conversion capabilities with a user-friendly GUI interface.

## New in Version 2.0

### Major Features Added
- Enhanced progress tracking with accurate status updates
- Improved AI optimization with better codec handling
- Multi-segment trimming capability
- Advanced audio analysis features

### Technical Improvements
- Fixed librosa import issues
- Improved FFmpeg integration
- Better error handling
- Updated dependencies

## Features

### Media Conversion
- Support for multiple formats:
  - Video: MP4, AVI, MOV, MKV, WebM, FLV
  - Audio: WAV, MP3, AAC, FLAC, OGG
  - Screen Recordings from Android, iPhone, macOS, Windows
- AI-optimized conversion options
- Customizable output settings

### Audio Analysis
- Detailed audio file analysis
- Silence detection
- Waveform visualization
- Advanced audio features extraction

### Trimming Features
- Simple and advanced trimming modes
- Multi-segment selection
- Visual waveform editing
- Real-time preview capability

### Batch Processing
- Process multiple files simultaneously
- Queue management
- Progress tracking
- Batch operation logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lostmind008/advanced-media-converter.git
cd advanced-media-converter
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- On macOS: `brew install ffmpeg`
- On Ubuntu/Debian: `sudo apt-get install ffmpeg`
- On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

Run the application:
```bash
python MediaConverter.py
```

## Requirements
- Python 3.7+
- FFmpeg
- Additional dependencies in requirements.txt

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## Acknowledgments
- FFmpeg team for the powerful media processing toolkit
- Python community for excellent libraries
- Contributors and users for feedback and suggestions
