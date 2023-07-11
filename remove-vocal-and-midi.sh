for mp3 in data/mp3/*.mp3; do
    echo "Processing $mp3"
    # remove vocals
    spleeter separate -p spleeter:2stems -o data/sep "$mp3"
    fname="${mp3%.*}"
    fname=`basename "$fname"`
    acc="data/sep/$fname/accompaniment.wav"
    mkdir -p data/midi/$fname
    basic-pitch data/midi/$fname "$acc"
done
