
# Prüfe, ob mit_rirs leer oder neu ist
if [ ! -d "mit_rirs" ] || [ -z "$(ls -A mit_rirs)" ]; then
mkdir -p mit_rirs

    curl -LsSf https://hf.co/cli/install.sh | bash
    hf download davidscripka/MIT_environmental_impulse_responses --repo-type=dataset

    # Kopiere Audiosamples aus dem Huggingface-Cache in den lokalen mit_rirs-Ordner
    HF_CACHE_DIR="${HOME}/.cache/huggingface/hub/datasets--davidscripka--MIT_environmental_impulse_responses"
    # Der Hash-Ordner in snapshots ist variabel, daher suchen wir ihn dynamisch
    SNAPSHOTS_DIR=$(ls -d "${HF_CACHE_DIR}/snapshots"/*/16khz 2>/dev/null | head -1)

    if [ -d "$SNAPSHOTS_DIR" ]; then
        echo "Kopiere Audiosamples aus ${SNAPSHOTS_DIR}..."
        # Kopiere alle Dateien/Symlinks mit ihren ursprünglichen Namen
        for audio_file in "$SNAPSHOTS_DIR"/*.wav; do
        if [ -f "$audio_file" ]; then
            filename=$(basename "$audio_file")
            cp "$audio_file" "mit_rirs/${filename}"
            echo "  Kopiert: ${filename}"
        fi
        done
        echo "Fertig! Audiosamples sind im Ordner 'mit_rirs' verfügbar."
    else
        echo "Warnung: Snapshots-Verzeichnis nicht gefunden: ${HF_CACHE_DIR}/snapshots/*/16khz"
    fi
fi

# Verzeichnis erstellen falls nicht vorhanden
if [ ! -d "audioset" ] && [ ! -d "audioset_16k" ]; then
    mkdir audioset
    
    # Datei herunterladen
    fname="bal_train09.tar"
    out_dir="audioset/${fname}"
    link="https://huggingface.co/datasets/agkphysics/AudioSet/resolve/196c0900867eff791b8f4d4be57db277e9a5b131/bal_train09.tar?download=true"
    
    wget -O "$out_dir" "$link"
    
    # Entpacken
    cd audioset && tar -xf bal_train09.tar
    cd ..
fi

# Output-Verzeichnis erstellen
output_dir="./audioset_16k"
if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"

    # FLAC zu WAV konvertieren (16kHz, 16-bit PCM)
    # Findet alle .flac Dateien und konvertiert sie
    find audioset/audio -type f -name "*.flac" | while read -r flac_file; do
        # Dateiname ohne Pfad und mit .wav Endung
        name=$(basename "$flac_file" .flac).wav
        
        # Konvertierung mit ffmpeg oder sox
        # Mit ffmpeg:
        ffmpeg -i "$flac_file" -ar 16000 -ac 1 -sample_fmt s16 "${output_dir}/${name}" -y 2>/dev/null
        
        # ODER mit sox (falls ffmpeg nicht verfügbar):
        # sox "$flac_file" -r 16000 -b 16 -c 1 "${output_dir}/${name}"
        
        echo "Konvertiert: $name"

        # audioset Verzeichnis kann gelöscht werden
        rm -rf audioset
    done
fi

if [ ! -d "fma" ] && [ ! -d "fma_16k" ]; then

    mkdir -p fma
    wget -O fma/fma_xs.zip https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip
    cd fma
    unzip -q fma_xs.zip
    cd ..
    mkdir -p fma_16k
    find fma/fma_small -type f -name "*.mp3" -exec bash -c 'ffmpeg -i "$0" -ar 16000 -ac 1 -sample_fmt s16 "fma_16k/$(basename "$0" .mp3).wav" -y' {} \;

    rm -rf fma
fi

if [ ! -d "negative_datasets" ]; then
    mkdir -p negative_datasets

    for fname in dinner_party.zip dinner_party_eval.zip no_speech.zip speech.zip; do
        wget -O "negative_datasets/$fname" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/$fname"
        unzip -q "negative_datasets/$fname" -d negative_datasets

        rm "negative_datasets/$fname"
    done
fi


echo "Fertig!"