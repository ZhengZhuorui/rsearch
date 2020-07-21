
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image

from rsearch_backend.CMH import CMH

import os
import random

class Encoder():
	def __init__(self, modelpath = './rsearch_backend/model/hashmodel.pth.tar'):
		self.model = CMH()
		self.optimizer = torch.optim.SGD([{'params': self.model.parameters(), 'initial_lr': 0.001}], lr = 0.01,  weight_decay = 0.01)
		if os.path.isfile(modelpath):
			checkpoint = torch.load(modelpath)
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			START_EPOCH = checkpoint['epoch']
			print("Checkpoint load finished")
		else:
			print("No checkpoint found")
		self.CNN = models.vgg16(pretrained = True)
		self.CNN.classifier = torch.nn.Sequential(*list(self.CNN.classifier.children())[:-3])
	
		self.model.eval()
		self.CNN.eval()
		if torch.cuda.is_available():
			self.model = self.model.cuda()
			self.CNN = self.CNN.cuda()

		self.worddict = self._buildDict()
		self.default_i = [0 for i in range (4096)]
		self.default_t = [0 for i in range(128)]


	def _buildDict(self):
		file = open('./rsearch_backend/model/Labels.txt')
		data = file.read().split('\n')[ : -1]
		words = {}
		i = 0
		for row in data:
			row = row.split(' ')
			for word in row:
				if word in words.keys():
					continue
				words[word] = i
				i = i + 1
		return words


	def textEmbedding(self, text):
		words = text.split(' ')
		vects = [0 for i in range(128)]
		for word in words:
			if word in self.worddict.keys():
				vects[self.worddict[word]] = 1
		return vects

	def openImage(self, filename):
		input_image = Image.open(filename)
		input_image = input_image.convert("RGB")
		preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		input_tensor = preprocess(input_image)
		return input_tensor


	def imageEmbedding(self, imagefile):
		imagelist = []
		imagelist.append(self.openImage(imagefile).numpy())
		imagedataset = TensorDataset(torch.FloatTensor(imagelist))
		dataloader = torch.utils.data.DataLoader(imagedataset, batch_size = 64)
		embeddings = []
		with torch.no_grad():
			for t, (input_batch) in enumerate(dataloader):
				input_batch = input_batch[0]
				if torch.cuda.is_available():
					input_batch = input_batch.cuda()
				output = self.CNN(input_batch)
				#print(len(output.data.cpu().numpy()))
				embeddings.extend(output.data.cpu().numpy())
				#print(embeddings[-1])
		return embeddings[0]
		#print(embeddings[0])


	def textEncoding(self, text):
		embedding = self.textEmbedding(text)
		embeddings = [embedding]
		t2idataset = TensorDataset(torch.FloatTensor(embeddings), torch.FloatTensor([self.default_i]), torch.FloatTensor([self.default_i]))
		t2idataloader = torch.utils.data.DataLoader(t2idataset, batch_size = 64, shuffle = True)
		with torch.no_grad():
			for ite, (query_data, input_pos, input_neg)in enumerate(t2idataloader):
				if torch.cuda.is_available():
					query_data = query_data.cuda()
					input_pos = input_pos.cuda()
					input_neg = input_neg.cuda()
				#query_data = Variable(query_data)
				#input_pos = Variable(input_pos)
				#input_neg = Variable(input_neg)
				vect = self.model(query_data, input_pos, input_neg, 't2i')	
		#print(vect.data.cpu().numpy()[0][0])	
		return vect.data.cpu().numpy()[0]



	def imageEncoding(self, imagepath):
		embedding = self.imageEmbedding(imagepath)
		embeddings = [embedding]
		i2tdataset = TensorDataset(torch.FloatTensor(embeddings), torch.FloatTensor([self.default_t]), torch.FloatTensor([self.default_t]))
		i2tdataloader = torch.utils.data.DataLoader(i2tdataset, batch_size = 64, shuffle = True)
		with torch.no_grad():
			for ite, (query_data, input_pos, input_neg)in enumerate(i2tdataloader):
				if torch.cuda.is_available():
					query_data = query_data.cuda()
					input_pos = input_pos.cuda()
					input_neg = input_neg.cuda()
				#query_data = Variable(query_data)
				#input_pos = Variable(input_pos)
				#input_neg = Variable(input_neg)
				vect = self.model(query_data, input_pos, input_neg, 'i2t')	
		#print(vect.data.cpu().numpy()[0][0])	
		return vect.data.cpu().numpy()[0]


if __name__ == "__main__":
	encoder = Encoder()
	print(encoder.textEncoding("airplane"))
	#print(encoder.imageEncoding("./Images/Image0.jpg"))
