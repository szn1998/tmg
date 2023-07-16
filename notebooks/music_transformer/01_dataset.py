# -*- coding: utf-8 -*-
import os
import math
import statistics
import random
from tqdm import tqdm
import TMIDIX

# !ln -sf ../../data/tmg-datasets/pitch/ dataset

# !wget 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz'
# !tar -xvf 'lmd_full.tar.gz'
# !rm 'lmd_full.tar.gz'

dataset_addr = "dataset"
# os.chdir(dataset_addr)
filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

random.shuffle(filez)

TMIDIX.Tegridy_Any_Pickle_File_Writer(filez, 'pickle/data')

#@title Load file list
filez = TMIDIX.Tegridy_Any_Pickle_File_Reader('pickle/data')

# # Process

START_FILE_NUMBER = 0
LAST_SAVED_BATCH_COUNT = 0

input_files_count = START_FILE_NUMBER
files_count = LAST_SAVED_BATCH_COUNT

melody_chords_f = []

stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for f in tqdm(filez[START_FILE_NUMBER:]):
    try:
        input_files_count += 1

        fn = os.path.basename(f)

        # Filtering out giant MIDIs
        file_size = os.path.getsize(f)

        if file_size < 250000:

          #=======================================================
          # START PROCESSING

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
            if min([e[1] for e in events_matrix1]) >= 0 and min([e[2] for e in events_matrix1]) > 0:
              
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

                if len([y for y in melody_chords if y[2] != 9]) > 12: # Filtering out tiny/bad MIDIs...

                  times = [y[0] for y in melody_chords[12:]]
                  avg_time = sum(times) / len(times)

                  times_list = list(set(times))

                  mode_dur = statistics.mode([y[1] for y in melody_chords if y[2] != 9])

                  instruments_list = list(set([y[2] for y in melody_chords]))
                  num_instr = len(instruments_list)

                  if instruments_list != [9]: # Filtering out bad MIDIs...
                    if avg_time < 64 and mode_dur < 64: # Filtering out bad MIDIs...
                      if 0 in times_list: # Filtering out (mono) melodies MIDIs
                        num_chords = len([y for y in melody_chords if y[0] != 0])
                        if num_chords > 600 and num_chords < (256 * 50):

                            #=======================================================
                            # FINAL PROCESSING
                            #=======================================================

                            # Break between compositions / Intro seq

                            if 9 in instruments_list:
                              drums_present = 3074 # Yes
                            else:
                              drums_present = 3073 # No

                            melody_chords_f.extend([3087, drums_present, 3075+melody_chords[0][2]])

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
                                
                              # Outro token
                              if (((num_chords // 50) * 50) - chords_count == 50) and time != 0:
                                melody_chords_f.extend([3072])

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

                              stats[m[2]] += 1

                            #=======================================================

                            # Processed files counter
                            files_count += 1

                            # Saving every 5000 processed files
                            if files_count % 5000 == 0:
                              print('SAVING !!!')
                              print('=' * 70)       
                              print('Saving processed files...')
                              print('=' * 70)
                              print('Data check:', min(melody_chords_f), '===', max(melody_chords_f), '===', len(list(set(melody_chords_f))), '===', len(melody_chords_f))
                              print('=' * 70)
                              print('Processed so far:', files_count, 'out of', input_files_count, '===', files_count / input_files_count, 'good files ratio')
                              print('=' * 70)
                              count = str(files_count)
                              TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, '/content/drive/MyDrive/LAKH_INTs_'+count)
                              melody_chords_f = []
                              print('=' * 70)
        
    except KeyboardInterrupt:
        print('Saving current progress and quitting...')
        break  

    except Exception as ex:
        print('WARNING !!!')
        print('=' * 70)
        print('Bad MIDI:', f)
        print('Error detected:', ex)
        print('=' * 70)
        continue

# Saving last processed files...
print('SAVING !!!')     
print('=' * 70)
print('Saving processed files...')
print('=' * 70)
print('Data check:', min(melody_chords_f), '===', max(melody_chords_f), '===', len(list(set(melody_chords_f))), '===', len(melody_chords_f))
print('=' * 70)
print('Processed so far:', files_count, 'out of', input_files_count, '===', files_count / input_files_count, 'good files ratio')
print('=' * 70)
count = str(files_count)
TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, 'chords/LAKH_INTs_'+count)

print('Instruments stats:')
print('=' * 70)
print('Piano:', stats[0])
print('Guitar:', stats[1])
print('Bass:', stats[2])
print('Violin:', stats[3])
print('Cello:', stats[4])
print('Harp:', stats[5])
print('Trumpet:', stats[6])
print('Sax:', stats[7])
print('Flute:', stats[8])
print('Drums:', stats[9])
print('Choir:', stats[10])
print('Organ:', stats[11])
print('=' * 70)

"""# (TEST INTS)"""

# @title Test INTs

train_data1 = melody_chords_f

print('Sample INTs', train_data1[:15])

out = train_data1[:200000]

if len(out) != 0:
    
    song = out
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
                                                        output_file_name = 'midi', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')

"""# Congrats! You did it! :)"""
