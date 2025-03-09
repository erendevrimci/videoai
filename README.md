# VideoAI Automation Pipeline

An automated pipeline for generating AI videos and publishing them to YouTube.

## Features

- Script generation using OpenAI
- Voice-over generation using ElevenLabs
- Automatic caption generation
- Video editing with clip selection based on script content
- YouTube title and description generation
- Automated YouTube uploads
- Multi-channel support

## Setup

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   YOUTUBE_API_KEY=your_youtube_key
   YOUTUBE_CLIENT_ID=your_client_id
   YOUTUBE_CLIENT_SECRET=your_client_secret
   ```

3. Place video clips in the `clips/` directory and define them in `clips/clips_label.md`

4. Add background music MP3 files to the `background_music/` directory

## Usage

### Running the Full Pipeline

```bash
# Run the complete pipeline for all channels
python main.py

# Run for a specific channel
python main.py --channel 1

# Run specific steps only
python main.py --steps script,voice,video

# Run with a custom delay between channels
python main.py --delay 120
```

### Running Individual Steps

```bash
# Generate script
python write_script.py

# Generate voice for a specific channel
python voice_over.py 1

# Generate captions
python captions.py

# Edit video
python video_edit.py

# Generate title/description
python write_title_desc.py

# Upload to YouTube
python upload_video.py
```

## Configuration

The system uses a centralized configuration system in `config.py`. You can modify settings there instead of changing the code directly.

Key configuration options:
- API settings (models, parameters)
- Channel-specific settings (voice IDs, YouTube credentials)
- File paths
- Video editing parameters

## Directory Structure

- `clips/` - Video clip files and metadata
- `voice/` - Generated voice files
- `background_music/` - Background music files
- `outputs/` - Channel-specific output directories
  - `channel_1/` - Channel 1 outputs
  - `channel_2/` - Channel 2 outputs
  - `channel_3/` - Channel 3 outputs

## Channels

The system supports multiple YouTube channels, each with its own:
- Voice configuration
- YouTube credentials
- Output files
