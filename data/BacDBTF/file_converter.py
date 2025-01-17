import h5py
import pickle
import random
import os
import pandas as pd
import numpy as np

def from_pkl_to_h5():
	pkl_path = 'deepTF_acnes_650M_embeddings.pkl'
	with open(pkl_path, 'rb') as pkl:
		embeddings = pickle.load(pkl)['protein_embs']

	with h5py.File('deepTF_acnes_650M_embeddings.h5',"w") as hf:
		for seq_id, embedding in embeddings.items():
			key = seq_id.split('|')[1]
			hf.create_dataset(key, data=embedding)

def generate_200val():
	# 1. extract randomly a protein for each family in the dataset
	df = pd.read_csv('BacDBTF_morethan50.tsv', sep='\t')
	families = set(df['Family'].to_list())

	dataset = []
	for family in families: 
		proteins = list(df.loc[df['Family']==family, 'UniprotID'])
		dataset.append(random.choice(proteins))

	# 2. extract the embeddings for the dataset
	h5_f = h5py.File('BacDBTF_morethan50_650M_embeddings.h5', 'r')
	embeddings_dict = {}
	for key in h5_f.keys(): 
		embeddings_dict[key] = h5_f[key][()]

	# 3. save the embeddings in an h5 format 
	with h5py.File('BacDBTF_val200_650M_embeddings.h5',"w") as hf:
		for seq_id, embedding in embeddings_dict.items():
			if seq_id.split('_')[0] in dataset:
				hf.create_dataset(seq_id, data=embedding)
	return dataset

def generate_training_set(dataset_val200):
	# 1. generate a fasta from the val200 dataset 
	df = pd.read_csv('BacDBTF_morethan50.tsv', sep='\t')
	# with open('val200.fasta', 'w') as fasta:
	# 	for uniprot in dataset_val200: 
	# 		seq = list(df.loc[df['UniprotID']==uniprot, 'Sequence'])[0]
	# 		fasta.write('>{}\n{}\n'.format(uniprot, seq))

	# if os.path.isdir('mmseqs') == False:
	# 	os.mkdir('mmseqs')
	# os.chdir('mmseqs')

	# 2. execute mmseqs
	# os.system('sh mmseqs.sh')
	# os.system('mmseqs convertalis targetDB targetDB resultDB resultDB.m8')

	# 2.1 get the proteins with less than pident of id with the queries
	results = pd.read_csv('mmseqs/resultDB.m8', sep='\t', header=None, index_col=False)
	uniprots_lessthan = []
	for i in range(len(results.axes[0])):
		uniprot = results.at[i, 0].split('_')[0]
		if uniprot in dataset_val200: 
			ident = results.at[i, 2]
			if float(ident) <= 0.5:
				uniprots_lessthan.append(results.at[i, 1].split('_')[0])
	print(len(uniprots_lessthan))

	# 3. generate fasta
	with open('BacDBTF_0.5id_train3600.fasta', 'w') as fasta:
		for uniprot in uniprots_lessthan: 
			seq = list(df.loc[df['UniprotID']==uniprot, 'Sequence'])[0]
			fasta.write('>{}\n{}\n'.format(uniprot, seq))


	# 3.1 retreive embeddings and save them in h5
	h5_f = h5py.File('BacDBTF_morethan50_650M_embeddings.h5', 'r')
	embeddings_dict = {}
	for key in h5_f.keys(): 
		embeddings_dict[key] = h5_f[key][()]

	with h5py.File('BacDBTF_0.5id_train3600_embeddings.h5',"w") as hf:
		for seq_id, embedding in embeddings_dict.items():
			if seq_id.split('_')[0] in uniprots_lessthan:
				hf.create_dataset(seq_id, data=embedding)


def remove_non_existing_families_in_train_from_val(): 
	# what families are present in the training dataset
	identifiers = []
	with open('BacDBTF_0.5id_train3600.fasta', 'r') as f: 
		for line in f: 
			if line.startswith('>'): 
				identifiers.append(line.strip().strip('>'))

	df = pd.read_csv('BacDBTF_morethan50.tsv', sep='\t')

	families = []
	for uniprot in identifiers: 
		families.append(list(df.loc[df['UniprotID'] == uniprot, 'Family'])[0])

	# remove the proteins from the validation dataset that do not have family in the training dataset
	# dictionary with uniprots and families for val38 proteins: 
	val38_dict = {}
	uniprots = []
	with open('val200.fasta', 'r') as f: 
		for line in f: 
			if line.startswith('>'): 
				uniprots.append(line.strip().strip('>'))
	val35_seq = {}
	for uniprot in uniprots: 
		val38_dict[uniprot] = list(df.loc[df['UniprotID'] == uniprot, 'Family'])[0]
		val35_seq[uniprot] = list(df.loc[df['UniprotID'] == uniprot, 'Sequence'])[0]

	val35_dict = val38_dict.copy()
	for uniprot, family in val38_dict.items():
		if family not in families: 
			del val35_dict[uniprot]

	# create new val35 files 
	with open('BacDBTF_val35.fasta', 'w') as outfile:
		for uniprot in val35_dict.keys():
			outfile.write('>{}\n{}\n'.format(uniprot, val35_seq[uniprot]))

	h5_f = h5py.File('BacDBTF_morethan50_650M_embeddings.h5', 'r')
	embeddings_dict = {}
	for key in h5_f.keys(): 
		embeddings_dict[key] = h5_f[key][()]

	# 3. save the embeddings in an h5 format 
	with h5py.File('BacDBTF_val35_650M_embeddings.h5',"w") as hf:
		for seq_id, embedding in embeddings_dict.items():
			if seq_id.split('_')[0] in val35_dict.keys():
				hf.create_dataset(seq_id, data=embedding)
			

def remove_acnes_from_training(): 
	# make a list with the acnes identifiers to remove 
	acnes_ids = []
	with open('deepTF_acnes.fasta', 'r') as fd: 
		for line in fd: 
			if line.startswith('>'): 
				uniprot = line.split('|')[1]
				acnes_ids.append(uniprot)

	# load the embeddings into a dictionary
	h5_f = h5py.File('BacDBTF_morethan50_650M_embeddings.h5', 'r')
	embeddings_dict = {}
	for key in h5_f.keys(): 
		embeddings_dict[key] = h5_f[key][()]

	embeddings_dict1 = embeddings_dict.copy()
	for k, v in embeddings_dict1.items(): 
		uniprot = k.split('_')[0]
		if uniprot in acnes_ids: 
			print(k)
			del embeddings_dict[k]
	
	with h5py.File('BacDBTF_morethan50_WOacnes_650M_embeddings.h5',"w") as hf:
		for seq_id, embedding in embeddings_dict.items():
			hf.create_dataset(seq_id, data=embedding)

if __name__=='__main__':
	# dataset_val200 = generate_200val()
	#dataset_val200 = []
	#with open('val200.fasta', 'r') as fasta: 
		#for line in fasta:
			#if line.startswith('>'):
			#	dataset_val200.append(line.strip('>').strip())

	#generate_training_set(dataset_val200)
	# remove_non_existing_families_in_train_from_val()

	remove_acnes_from_training()

