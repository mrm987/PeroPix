# PeroPix

> Generate hundreds of chatbot character assets in a day

## Dramatically reduce your asset creation and censoring time

The most time-consuming part of chatbot creation is **image generation**.

Especially generating emotion sets and H assets per character, then censoring them takes considerable time.

**PeroPix** generates pre-configured emotion/H sets all at once and automates the censoring process.

## Key Features

### Slot Mode: The Core of Mass Production
**Add unlimited slots with different prompts and generate everything at once.**

- Slot 1: `happy, smile`
- Slot 2: `sad, tears`
- Slot 3: `angry, shouting`
- ...

Set up prompts for each emotion in slots, then generate to create all assets for a character in one batch.

### Censor Mode: Auto Detection + One-Click Fix
No need to manually check hundreds of generated images.

Drop all images into the folder > Click [Auto Censor] > Done!

- **One-Click Auto Censoring**: YOLO model scans and automatically censors images.
- **Easy Manual Touch-ups**: Review the full list and easily fix any missed areas.

### NAI + Local, One App
PeroPix supports both NAI and Local (Stable Diffusion), switchable via tabs.

- **NovelAI**: Character Reference, Vibe Transfer, Inpaint - 100% identical results to the web.
- **Local Generation**: SDXL + LoRA, upscaling support.

Same slot/censor workflow works for both.

### Other Features
- Tag autocomplete (2.42M Danbooru DB)

## Installation

Download ZIP from [Releases](https://github.com/mrm987/PeroPix/releases) → Extract → Run executable

**Executables**
- **Windows**: `PeroPix.bat`
- **macOS**: `PeroPix.command`

## Usage

```
1. Settings → Enter NAI API Token
2. Write prompts
3. Click Queue
```

Generated files are automatically saved to the `outputs/` folder.

## System Requirements

Just need a NovelAI subscription. (Windows 10/11 or macOS)

For local generation, NVIDIA GPU with 12GB+ VRAM is recommended.

## Troubleshooting

### Where are the logs?
Logs are stored in the `logs/` folder inside the app directory.
Log files are named `peropix_YYYYMMDD.log` and contain detailed error information.

### App won't start
1. Check if the `logs/` folder exists and review recent log files
2. Try deleting `config.json` and restarting
3. Ensure no other application is using port 7860

### NAI API errors
- Verify your API token is correct (Settings → NAI API Token)
- Check your NovelAI subscription status
- Ensure you have enough Anlas

## Credits

Tag Autocomplete: [Danbooru Tag Database](https://github.com/DraconicDragon/dbr-e621-lists-archive) (2026-01-01)

Auto Censoring: [Anime NSFW Detection (ntd11)](https://civitai.com/models/1313556/anime-nsfw-detectionadetailer-all-in-one)

---

MIT License
