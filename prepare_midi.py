from midi_signals.midi import generateDataset
import argparse

parser = argparse.ArgumentParser(description='Create a dataset from MIDI files')

parser.add_argument('directory', type=unicode, help="the directory containing the MIDI files")
parser.add_argument('output', type=unicode, help="the output file (.pkl)")
args = parser.parse_args()

generateDataset(args.directory, args.output)