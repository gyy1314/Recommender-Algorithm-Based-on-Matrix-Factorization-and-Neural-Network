import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils
from MF_model import MFModel


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epo
	type=int,
	default=128,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

device = 'cpu'

############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all(model = config.model)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True,config.model)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if config.model == 'MF':
	model = MFModel(64,
					user_num,item_num,
					0.0005,
					0.007,
					40)
else:
	if config.model == 'NeuMF-pre':
		assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
		assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
		GMF_model = torch.load(config.GMF_model_path)
		MLP_model = torch.load(config.MLP_model_path)
	else:
		GMF_model = None
		MLP_model = None

	model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
							args.dropout, config.model, GMF_model, MLP_model)
	model.to(device)
	loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)


# ########################### TRAINING #####################################
if __name__=='__main__':
	count, best_hr = 0, 0
	for epoch in range(args.epochs):

		start_time = time.time()
		train_loader.dataset.ng_sample()
		if model == 'MF':
			batch = []
			for user, item, label in train_loader:
				for u,i,l in zip(user.tolist(),item.tolist(),label.tolist()):
					batch.append([u,i,l])
			model.train_step(batch)
		else:
			model.train() # Enable dropout (if have).
			for user, item, label in train_loader:
				user = user.to(device)
				item = item.to(device)
				label = label.float().to(device)
				model.zero_grad()
				prediction = model(user, item)
				loss = loss_function(prediction, label)
				loss.backward()
				optimizer.step()
				count += 1
			model.eval()
		HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
		if model == 'MF':
			tmp_HR = 0.65
			if HR >= tmp_HR:
				np.savetxt('./user_64_128.txt',model._user_factors,fmt = '%.6f',delimiter = ',', newline = '\n')
				np.savetxt('./user_bias_64_128.txt',model._user_bias,fmt = '%.6f',delimiter = ',', newline = '\n')
				np.savetxt('./item_64_128.txt',model._item_factors,fmt = '%.6f',delimiter = ',', newline = '\n')
				np.savetxt('./item_bias_64_128.txt',model._item_bias,fmt = '%.6f',delimiter = ',', newline = '\n')
				np.savetxt('./gobal_bias_64_128.txt',[model._global_bias],fmt = '%.6f',delimiter = ',', newline = '\n')
				tmp_HR = HR
		print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
		else:
			if HR > best_hr:
				best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
				if args.out:
					if not os.path.exists('./model1'):
						os.mkdir('./model1')
					torch.save(model, 
						'{}{}.pth'.format('./model1', config.model))

	print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
										best_epoch, best_hr, best_ndcg))
