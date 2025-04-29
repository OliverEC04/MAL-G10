#!/usr/bin/env python3

"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.

Shakespeare output:
	length of dataset in characters:  1115394
	all the unique characters:
	 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
	vocab size: 65
	train has 1003854 tokens
	val has 111540 tokens

HOML:
	length of dataset in characters: 1,052,423
	all the unique characters: 

	 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz{|}~©¬°±²³·½×àáçéöüıšŷ̇ΘΣαβγδεζηθλμσφχϕϵ‐–—‖’“”•…′ℒℓ←→∂∇∈∑−∞∣∥∧∨∫∼≈≠≤≥⊕⊗⊘⋮⋯─│└├�
	 vocab size: 172
	 train has 947,180 tokens
	 val has 105,243 tokens

MMLS:
	length of dataset in characters: 56,876
	all the unique characters: 

	 ()+,-./0123456789:<=>?ABCDEFGHIJKLMNOPQRSTUVW_abcdefghijklmnoprstuvxyz|±ÉØåæéø–’ℎℝℳ∈−∙√∞∠∧∶≤≥𝐀𝐈𝐛𝐴𝐵𝐶𝐷𝐹𝐼𝑃𝑄𝑅𝑉𝑍𝑎𝑏𝑐𝑑𝑒𝑓𝑔𝑖𝑗𝑘𝑚𝑟𝑡𝑣𝑤𝑥𝑦𝑧𝑨𝑿𝒃𝒙𝛳𝜃𝜇𝜋𝜑𝜔
	vocab size: 141
	train has 51,188 tokens
	val has 5,688 tokens
"""

import os
import pickle
import requests
import numpy as np

def Err(msg):
	print(f"ERROR: {msg}")
	exit(-1)

def Prepare():
	# download the tiny shakespeare dataset
	#input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

	root_dir = "./"
	input_file_path = os.path.join(root_dir, "input.txt")
	
	if not os.path.exists(input_file_path):
		Err(f"missing file {input_file_path}")
		#data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
		#with open(input_file_path, 'w') as f:
		#    f.write(requests.get(data_url).text)

	with open(input_file_path, 'r') as f:
		data = f.read()
	 
	print(f"length of dataset in characters: {len(data):,}")

	# get all the unique characters that occur in this text
	chars = sorted(list(set(data)))
	vocab_size = len(chars)
	print("all the unique characters:", ''.join(chars))
	print(f"vocab size: {vocab_size:,}")

	# create a mapping from characters to integers
	stoi = { ch:i for i,ch in enumerate(chars) }
	itos = { i:ch for i,ch in enumerate(chars) }
	def encode(s):
		return [stoi[c] for c in s] # encoder: take a string, output a list of integers
	def decode(l):
		return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

	# create the train and test splits
	n = len(data)
	train_data = data[:int(n*0.9)]
	val_data = data[int(n*0.9):]

	# encode both to integers
	train_ids = encode(train_data)
	val_ids = encode(val_data)
	print(f"train has {len(train_ids):,} tokens")
	print(f"val has {len(val_ids):,} tokens")

	# export to bin files
	train_ids = np.array(train_ids, dtype=np.uint16)
	val_ids   = np.array(val_ids, dtype=np.uint16)
	train_ids.tofile(os.path.join(root_dir, 'train.bin'))
	val_ids.tofile  (os.path.join(root_dir, 'val.bin'))

	# save the meta information as well, to help us encode/decode later
	meta = {
		'vocab_size': vocab_size,
		'itos': itos,
		'stoi': stoi,
	}
	with open(os.path.join(root_dir, 'meta.pkl'), 'wb') as f:
		pickle.dump(meta, f)

if __name__=='__main__':
	try:
		Prepare()
	except Exception as ex:
		#Diagnostics.PrettyPrintTraceback(ex)
		Err(ex)
		exit(-1)
