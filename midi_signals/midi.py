from mido import Message, MidiFile, MidiTrack
import mido
import numpy as np
import pickle
import os

def encodeMessage(msg):
    if msg.type == "note_on" or msg.type == "note_off":
        if msg.note >= 21 and msg.note <= 21 + 87:
            note_index = msg.note - 21
            if msg.type == "note_on":
                if msg.velocity == 0:
                    return (1 + note_index * 2 + 1, 0)
                else:
                    return (1 + note_index * 2, float(msg.velocity) / 127.0)
            if msg.type == "note_off":
                return (1 + note_index * 2 + 1, 0)
    if msg.type == "control_change" and msg.control == 64:
        return (0, 0)

def midi2Messages(filename):
    mid = mido.MidiFile(filename)
    tempo = 500000 # 60 bpm
    current_time = 0
    t_last_emit = None
    pedal_value = 0
    items = []
    for msg in mid:
        current_time += msg.time
        if not msg.is_meta:
            # Note or pedal
            if msg.type == "control_change" and msg.control == 64:
                pedal_value = float(msg.value) / 127.0

            c = encodeMessage(msg)
            if c is not None:
                if t_last_emit is None: t_last_emit = current_time
                dt = current_time - t_last_emit
                t_last_emit = current_time
                c = c + (pedal_value, dt,)
                items.append(c)
        else:
            if msg.type == "set_tempo":
                tempo = msg.tempo
    return items

# Note vector:
# [ DELTA_TIME,
#   CURRENT_PEDAL_VALUE
#   NOTE_VELOCITY,
#   IS_PEDAL_CHANGE,
#   IS_NOTE_0_ON, IS_NOTE_0_OFF, IS_NOTE_1_ON, IS_NOTE_1_OFF, ... ]
# size: 180

def encodedMessageToVector((note, velocity, pedal_value, dt)):
    result = np.zeros(180, dtype = np.float32)
    # Encode note
    result[0] = dt
    result[1] = pedal_value
    result[2] = velocity
    result[3 + note] = 1
    return result

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(1e-6 + a) / temperature
    a = np.exp(a) / np.sum(np.exp(a)) * 0.99999
    return np.argmax(np.random.multinomial(1, a, 1))

def vectorToEncodedMessage(v, temperature = 1.0):
    note = sample(v[3:], temperature)
    return (int(note), float(v[2]), float(v[1]), float(v[0]))

def midiClamp(x):
    x = int(x)
    if x < 0: return 0
    if x > 127: return 127
    return x

def convertEncodedMessagesToMIDI(msgs, filename):
    from mido import Message, MidiFile, MidiTrack

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 96

    for (note, velocity, pedal_value, dt) in msgs:
        time = int(mido.second2tick(dt, mid.ticks_per_beat, 500000))
        if note == 0:
            track.append(Message('control_change', control=64, value = midiClamp(pedal_value * 127.0), time=time))
        else:
            note -= 1
            if note % 2 == 1:
                track.append(Message('note_on', note=note / 2 + 21, velocity=0, time=time))
            else:
                track.append(Message('note_on', note=note / 2 + 21, velocity = midiClamp(velocity * 127.0), time=time))

    mid.save(filename)

def generateDataset(root_directory, output_file = "midi_dataset.pkl"):
    all_sequence = []

    for root, dirs, files in os.walk(root_directory):
        for f in files:
            path = os.path.join(root, f)
            if os.path.splitext(path)[1].lower() == ".mid":
                print path
                all_sequence += midi2Messages(path)

    print "Writing output..."

    with open(output_file, "wb") as f:
        pickle.dump(all_sequence, f)

def test(input_midi, output_midi):
    # This goes through the encoding / decoding process,
    # output_midi should sound identical as input_midi
    items = midi2Messages(input_midi)

    ritems = []

    data = np.zeros((len(items), 180))

    for i, item in enumerate(items):
        data[i,:] = encodedMessageToVector(item)
        v = vectorToEncodedMessage(data[i,:])
        ritems.append(v)

    convertEncodedMessagesToMIDI(ritems, output_midi)


