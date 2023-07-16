# # Generate Music

# # Imports

# +
import os
import pickle
import random
import secrets
import statistics
from time import time
import tqdm
import torch
import TMIDIX
from x_transformer import TransformerWrapper, Decoder, AutoregressiveWrapper
import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from midi2audio import FluidSynth
from IPython.display import Audio, display
# -

# # Utils

# +
import shutil

def divider():
    term_size = shutil.get_terminal_size().columns
    print('-' * term_size)


# -

divider()

model_checkpoint = "models/model_checkpoint_28699_steps_0.0504_loss_0.9873_acc.pth" 
SEQ_LEN = 2048

# !file $model_checkpoint

model = TransformerWrapper(
    num_tokens = 3088,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 32, heads = 8)
)

model = AutoregressiveWrapper(model)

model = torch.nn.DataParallel(model)

model.cuda()

model.load_state_dict(torch.load(model_checkpoint))

model.eval()

summary(model)

# # Plot Token Embeddings
#

tok_emb = model.module.net.token_emb.emb.weight.detach().cpu().tolist()

tok_emb1 = []

for t in tok_emb:
    tok_emb1.append([abs(statistics.median(t))])

cos_sim = metrics.pairwise_distances(
   tok_emb1, metric='euclidean'
)
plt.figure(figsize=(7, 7))
plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
plt.xlabel("Position")
plt.ylabel("Position")
plt.tight_layout()
plt.plot()
plt.savefig("log/Music-Transformer-Small-Tokens-Embeddings-Plot.png", bbox_inches="tight")

# drums_present_or_not = True 
# first_note_instrument = "Flute" 
# sangjee mod
drums_present_or_not = False
first_note_instrument = "Piano" 

number_of_tokens_tp_generate = 510 
number_of_batches_to_generate = 4 
temperature = 0.9 

velocities_map = [80, 80, 70, 100, 90, 80, 100, 100, 100, 90, 110, 100]

if drums_present_or_not:
  drumsp = 3074 # Yes
else:
  drumsp = 3073 # No

# +
# instruments_list = ["Piano", "Guitar", "Bass", "Violin", "Cello", "Harp", "Trumpet", "Sax", "Flute", 'Drums', "Choir", "Organ"]

# sangjee mod
instruments_list = ['Piano']
first_note_instrument_number = instruments_list.index(first_note_instrument)
# -

outy = [3087, drumsp, 3075+first_note_instrument_number]

inp = [outy] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()

out = model.module.generate(inp, 
                      number_of_tokens_tp_generate, 
                      temperature=temperature, 
                      return_prime=True, 
                      verbose=True)

out0 = out.tolist()

for i in range(number_of_batches_to_generate):

  print('=' * 70)
  print('Batch #', i)
  print('=' * 70)

  out1 = out0[i]

  print('Sample INTs', out1[:12])
  print('=' * 70)

  if len(out1) != 0:
    
      song = out1
      song_f = []

      time = 0
      dur = 0
      vel = 90
      pitch = 0
      channel = 0
                      
      for ss in song:
        
        if ss > 0 and ss < 256:

            time += ss * 8
          
        if ss >= 256 and ss < 1280:
            
            dur = ((ss-256) // 8) * 32
            vel = (((ss-256) % 8)+1) * 15
            
        if ss >= 1280 and ss < 2816:
            channel = (ss-1280) // 128
            pitch = (ss-1280) % 128

            song_f.append(['note', time, dur, channel, pitch, vel ])

      detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Allegro Music Transformer',  
                                                          output_file_name = 'results/composition_'+str(i), 
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                          number_of_ticks_per_quarter=500)


      print('=' * 70)
      print('Displaying resulting composition...')
      print('=' * 70)

      fname = 'results/composition_'+str(i)

      x = []
      y =[]
      c = []

      colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

      for s in song_f:
        x.append(s[1] / 1000)
        y.append(s[4])
        c.append(colors[s[3]])

      FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
      display(Audio(str(fname + '.wav'), rate=16000))

      plt.figure(figsize=(14,5))
      ax=plt.axes(title=fname)
      ax.set_facecolor('black')

      plt.scatter(x,y, c=c)
      plt.xlabel("Time")
      plt.ylabel("Pitch")
      plt.show()

select_seed_MIDI = "seed-biwang-1" 
full_path_to_custom_seed_MIDI = ""

if full_path_to_custom_seed_MIDI == '':
  f = 'seeds/'+select_seed_MIDI+'.mid'
else:
  f = full_path_to_custom_seed_MIDI

print('Allegro Music Transformer Seed MIDI Loader')
print('Loading seed MIDI...')
print('File:', f)

melody_chords_f = []

# Convering MIDI to ms score with MIDI.py module
score = TMIDIX.midi2ms_score(open(f, 'rb').read())

# INSTRUMENTS CONVERSION CYCLE
events_matrix = []
itrack = 1
patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

patch_map = [
            [0, 1, 2, 3, 4, 5, 6, 7], # Piano 
            [24, 25, 26, 27, 28, 29, 30], # Guitar
            [32, 33, 34, 35, 36, 37, 38, 39], # Bass
            [40, 41], # Violin
            [42, 43], # Cello
            [46], # Harp
            [56, 57, 58, 59, 60], # Trumpet
            [64, 65, 66, 67, 68, 69, 70, 71], # Sax
            [72, 73, 74, 75, 76, 77, 78], # Flute
            [-1], # Drums
            [52, 53], # Choir
            [16, 17, 18, 19, 20] # Organ
            ]

while itrack < len(score):
    for event in score[itrack]:         
        if event[0] == 'note' or event[0] == 'patch_change':
            events_matrix.append(event)
    itrack += 1

events_matrix.sort(key=lambda x: x[1])

events_matrix1 = []

for event in events_matrix:
        if event[0] == 'patch_change':
            patches[event[2]] = event[3]

        if event[0] == 'note':
            event.extend([patches[event[3]]])
            once = False
            
            for p in patch_map:
                if event[6] in p and event[3] != 9: # Except the drums
                    event[3] = patch_map.index(p)
                    once = True
                    
            if not once and event[3] != 9: # Except the drums
                event[3] = 15 # All other instruments/patches channel
                event[5] = max(80, event[5])
                
            if event[3] < 12: # We won't write chans 12-16 for now...
                events_matrix1.append(event)

if len(events_matrix1) > 0:           
  
    
    #=======================================================
    # PRE-PROCESSING

    # checking number of instruments in a composition
    instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))

    if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:

      # recalculating timings
      for e in events_matrix1:
          e[1] = int(e[1] / 8) # Max 2 seconds for start-times
          e[2] = int(e[2] / 32) # Max 4 seconds for durations

      # Sorting by pitch, then by start-time
      events_matrix1.sort(key=lambda x: x[4], reverse=True)
      events_matrix1.sort(key=lambda x: x[1])

      #=======================================================
      # FINAL PRE-PROCESSING

      melody_chords = []

      pe = events_matrix1[0]

      for e in events_matrix1:

          # Cliping all values...
          time = max(0, min(255, e[1]-pe[1]))             
          dur = max(1, min(127, e[2]))
          cha = max(0, min(11, e[3]))
          ptc = max(1, min(127, e[4]))

          # Calculating octo-velocity
          vel = max(8, min(127, e[5]))
          velocity = round(vel / 15)-1

          # Writing final note 
          melody_chords.append([time, dur, cha, ptc, velocity])

          pe = e

      times = [y[0] for y in melody_chords[12:]]
      avg_time = sum(times) / len(times)

      times_list = list(set(times))

      mode_dur = statistics.mode([y[1] for y in melody_chords if y[2] != 9])

      instruments_list = list(set([y[2] for y in melody_chords]))
      num_instr = len(instruments_list)
      
      #=======================================================

      # TOTAL DICTIONARY SIZE 3087+1=3088

      #=======================================================
      # MAIN PROCESSING CYCLE
      #=======================================================

      chords_count = 0

      melody_chords_f.extend([2816]) # Zero chords count

      if melody_chords[0][0] == 0:
        melody_chords_f.extend([0]) # Zero time, if present

      for m in melody_chords:
        
        time = m[0]

        # Chords counter token
        if chords_count % 50 == 0 and chords_count != 0 and time != 0:
          melody_chords_f.extend([2816+min(255, ((chords_count // 50)))])
          
        if time != 0:
          chords_count += 1                                

        # WRITING EACH NOTE HERE
        dur_vel = (m[1] * 8) + m[4]
        cha_ptc = (m[2] * 128) + m[3]

        if time != 0:
            melody_chords_f.extend([time, dur_vel+256, cha_ptc+1280])

        else:
            melody_chords_f.extend([dur_vel+256, cha_ptc+1280])

song = melody_chords_f

song_f = []

time = 0
dur = 0
vel = 90
pitch = 0
channel = 0

for ss in song:
  
  if ss > 0 and ss < 256:

      time += ss * 8
    
  if ss >= 256 and ss < 1280:
      
      dur = ((ss-256) // 8) * 32
      vel = (((ss-256) % 8)+1) * 15
      
  if ss >= 1280 and ss < 2816:
      channel = (ss-1280) // 128
      pitch = (ss-1280) % 128

      song_f.append(['note', time, dur, channel, pitch, vel ])

detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                      output_signature = 'Allegro Music Transformer',  
                                                      output_file_name = 'seed-composition/seed',
                                                      track_name='Project Los Angeles',
                                                      list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                      number_of_ticks_per_quarter=500)

print('Composition stats:')
print('Composition has', len([y for y in melody_chords_f if y >= 1280 and y < 2816]), 'notes')
print('Composition has', len(melody_chords_f), 'tokens')

fname = 'seed-composition/seed'

x = []
y =[]
c = []

colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

for s in song_f:
  x.append(s[1] / 1000)
  y.append(s[4])
  c.append(colors[s[3]])

FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
display(Audio(str(fname + '.wav'), rate=16000))

plt.figure(figsize=(14,5))
ax=plt.axes(title=fname)

plt.scatter(x,y, c=c)
plt.xlabel("Time")
plt.ylabel("Pitch")
plt.show()

# +
number_of_prime_tokens = 1000 
number_of_tokens_to_generate = 520  # how long the song should be generated
number_of_batches_to_generate = 10 # how many song to generate 
temperature = 0.999

try_to_generate_outro = False

include_prime_tokens_in_generated_output = True
allow_model_to_stop_generation_if_needed = False 

if allow_model_to_stop_generation_if_needed:
  min_stop_token = 3087
else:
  min_stop_token = None

outy = melody_chords_f[:number_of_prime_tokens]

if try_to_generate_outro:
  outy.extend([3072])


inp = [outy] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()


out = model.module.generate(inp, 
                      number_of_tokens_to_generate, 
                      temperature=temperature, 
                      return_prime=include_prime_tokens_in_generated_output, 
                      eos_token=min_stop_token, 
                      verbose=True)
out0 = out.tolist()
for i in range(number_of_batches_to_generate):

  print('=' * 70)
  print('Batch #', i)
  print('=' * 70)

  out1 = out0[i]

  print('Sample INTs', out1[:12])
  print('=' * 70)

  if len(out) != 0:
      
      song = out1
      song_f = []

      time = 0
      dur = 0
      vel = 90
      pitch = 0
      channel = 0
                      
      for ss in song:
        
        if ss > 0 and ss < 256:

            time += ss * 8
          
        if ss >= 256 and ss < 1280:
            
            dur = ((ss-256) // 8) * 32
            vel = (((ss-256) % 8)+1) * 15
            
        if ss >= 1280 and ss < 2816:
            channel = (ss-1280) // 128
            pitch = (ss-1280) % 128

            song_f.append(['note', time, dur, channel, pitch, vel ])

      detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Sangjee Dondrub <sangjeedondrub@live.com>',  
                                                          output_file_name = 'results/composition__'+str(i), 
                                                          track_name='AI Tibetan',
                                                          list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                          number_of_ticks_per_quarter=500)
      print('Displaying resulting composition...')
      fname = 'results/composition__'+str(i)

      x = []
      y =[]
      c = []

      colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

      for s in song_f:
        x.append(s[1] / 1000)
        y.append(s[4])
        c.append(colors[s[3]])

      FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
      display(Audio(str(fname + '.wav'), rate=16000))

      plt.figure(figsize=(14,5))
      ax=plt.axes(title=fname)
      ax.set_facecolor('black')

      plt.scatter(x,y, c=c)
      plt.xlabel("Time")
      plt.ylabel("Pitch")
      plt.show()