# # Data Collection

# + language="bash"
#
# ln -sf ../data .

# + language="bash"
#
# for mp3 in `find data/tmg-datasets/raw/*.mp3`
# do
#     fname="${mp3%.*}"
#     fname=`basename $fname`
#     ffmpeg -i $mp3 -ar 16000 -ac 1 -c:a pcm_s16le data/tmg-datasets/raw-wav/${fname}.wav
# done
# -

from tikit.utils.files import Path
from tikit.utils.manipulator import flatten2d

seps = flatten2d([list(y.glob('*.wav')) for y in [ x for x in list(Path('data/sep').glob('*')) if x.is_dir()]])

# +
ACC_DIR = Path('data/tmg-datasets/accompaniment')
VOCAL_DIR = Path('data/tmg-datasets/vocal')

for d in [ACC_DIR, VOCAL_DIR]:
    d.mkdir(exist_ok=True)
# -

for sep in seps:
    new_name = sep.parts[2]
    if sep.name == 'vocals.wav':
        new_v = VOCAL_DIR / f"{new_name}.wav"
        sep.copy(new_v)
    elif sep.name == 'accompaniment.wav':
        new_v = ACC_DIR / f"{new_name}.wav"
        sep.copy(new_v)

mids = flatten2d([list(y.glob('*.mid')) for y in [ x for x in list(Path('data/midi').glob('*')) if x.is_dir()]])

# +
EXP_DIR = Path("data/tmg-datasets/pitch")
EXP_DIR.mkdir(exist_ok=True)

for mid in mids:
    new_mid_name = mid.parts[2]
    mid.copy(EXP_DIR / f'{new_mid_name}.mid')

# +
EXP_DIR = Path("data/tmg-datasets/raw")
EXP_DIR.mkdir(exist_ok=True)

for mid in mp3:
    new_mid_name = mid.parts[2]
    mid.copy(EXP_DIR / f'{new_mid_name}')
# -

mp3 = [ x for x in list(Path('data/mp3').glob('*.mp3')) if x.is_file()]

mp3

ms = [x.parts[2].replace('.mid', '') for x in mids]

ds = [x.parts[2].replace('.mp3', '') for x in mp3]

len(ms), len(ds)

len(list(set(ms)))
