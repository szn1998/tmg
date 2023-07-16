# +
model.eval()

inp = [random.choice(val_dataset)[:512].tolist()]

inp = torch.LongTensor(inp).cuda()

print(inp)

sample = model.module.generate(inp, 512, temperature=0.9, return_prime=True)

print(sample)

# @title Convert tp MIDI

train_data1 = sample[0].tolist()

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
                                                        output_signature = 'Allegro Music Transformer',  
                                                        output_file_name = '/content/Allegro-Music-Transformer-Composition', 
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)

    print('Done!')

"""# Congrats! You did it! :)"""
