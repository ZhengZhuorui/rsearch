import torch
import torch.nn as nn


class CMH(nn.Module):
	def __init__(self, image_dim = 4096, text_dim = 128, hidden_dim = 4096, output_dim = 128, weight_decay = 0.01, beta = 2, gamma = 0.1, learning_rate = 0.001, param = None):
		super(CMH,self).__init__()
		self.image_dim = image_dim
		self.text_dim = text_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.weight_decay = weight_decay
		self.learning_rate = learning_rate
		self.beta = beta
		self.gamma = gamma
		self.param = param

		self.igenerator = nn.Sequential(
				nn.Linear(self.image_dim, self.hidden_dim),
				nn.Tanh(),
				nn.Linear(self.hidden_dim, self.output_dim),
				nn.Sigmoid()
			)

		self.tgenerator = nn.Sequential(
				nn.Linear(self.text_dim, self.hidden_dim),
				nn.Tanh(),
				nn.Linear(self.hidden_dim, self.output_dim),
				nn.Sigmoid()
			)
		self._initialize_weights()

	def _initialize_weights(self):
		for y, m in enumerate(self.modules()):
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.1)
				m.bias.data.zero_()

	def forward(self, query, input_pos, input_neg, flag):
		self.flag = flag
		#print(query.size())
		if flag == 'i2t':
			self.query_sig = self.igenerator(query)
			self.query_hash = torch.add(self.query_sig, 0.5).type('torch.IntTensor')
			self.pos_sig = self.tgenerator(input_pos)
			self.pos_hash = torch.add(self.pos_sig, 0.5).type('torch.IntTensor')
			self.neg_sig = self.tgenerator(input_neg)
			self.neg_hash = torch.add(self.neg_sig, 0.5).type('torch.IntTensor')

		else:
			self.query_sig = self.tgenerator(query)
			self.query_hash = torch.add(self.query_sig, 0.5).type('torch.IntTensor')
			self.pos_sig = self.igenerator(input_pos)
			self.pos_hash = torch.add(self.pos_sig, 0.5).type('torch.IntTensor')
			self.neg_sig = self.igenerator(input_neg)
			self.neg_hash = torch.add(self.neg_sig, 0.5).type('torch.IntTensor')	

		return self.query_sig 

	def getloss(self):
		#size = list(self.query_sig.size())[0]
		pred_distance = torch.sum(torch.pow(self.query_sig - self.pos_sig, 2), 1)
		hash_score = torch.sum(torch.eq(self.query_hash, self.pos_hash).type('torch.FloatTensor'), 1)
		pred_i2t_neg_distance = torch.sum(torch.pow(self.query_sig - self.neg_sig, 2), 1)
		
		#svm_loss
		#i2t_cross = torch.clamp(self.beta + pred_distance - pred_i2t_neg_distance, min = 0.0 )
		#i2t_loss = torch.mean(i2t_cross) # with L2 regularization as weight_decay in optimizer
		#i2t_reward = torch.sigmoid(i2t_cross).view(-1)
		#log_loss
		#i2t_cross = pred_i2t_neg_distance - pred_distance
		#i2t_cross = torch.clamp(self.beta + pred_distance - pred_i2t_neg_distance)
		i2t_cross = pred_i2t_neg_distance - pred_distance
		i2t_loss = - torch.mean(torch.log(torch.sigmoid(i2t_cross)))
		i2t_reward = torch.log(torch.sigmoid(i2t_cross)).view(-1)

		return i2t_loss

if __name__ == '__main__':
	model = CMH()
