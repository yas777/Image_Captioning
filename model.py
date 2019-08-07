import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import matplotlib.pyplot as plt
import numpy as np
import math

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        """Load the pretrained ResNet-52 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2] # Remove the last fc and pooling layers
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # self.fine_tune()

    def forward(self, images):
        """
        images: (batch_size, 3, image_size, image_size)
        """
        features = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        features = self.adaptive_pool(features)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return features

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        """
        encoder_dim: feature size of encoded images
        hidden_dim: size of hidden state of RNN
        attention_dim: size of the attention network
        """

        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out : encoded images, (batch_size, num_pixels, encoder_dim)
        decoder_hidden: previous decoder output, (batch_size, hidden_dim)
        """
        # print("-"*8 + "Attention" + "-"*8)
        # print("[ encoder_out size ] : " + str(encoder_out.size()))
        # print("[ decoder_hidden size ] : "+ str(decoder_hidden.size()))
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha



class DecoderRNN(nn.Module):

    def __init__(self, ntokens, embed_dim, hidden_dim, idx2word, word2idx, attention_dim = 512, encoder_dim = 2048, max_seg_length=20, dropout=0.5, fine_tune=True):
        super(DecoderRNN, self).__init__()
        self.encoder = nn.Embedding(ntokens, embed_dim)
        self.decoder = nn.Linear(hidden_dim, ntokens)

        # self.lstm = getattr(nn, rnn_type)(inp_dim, hidden_dim)
        self.att = Attention(encoder_dim, hidden_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.init_h = nn.Linear(encoder_dim, hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, hidden_dim)  # linear layer to find initial cell state of lstm cell 
        self.f_beta = nn.Linear(hidden_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.ntokens = ntokens
        self.max_seg_length = max_seg_length
        self.idx2word = idx2word
        self.word2idx = word2idx
        if not fine_tune:
            for p in self.encoder.parameters():
                p.requires_grad = fine_tune
        
    def init_weights(self):
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out):
        # encoder_out : B*num_pixels*encoder_dim
        mean_encoder_out = encoder_out.mean(dim = 1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, input, encoder_out, lengths):
        """
        processing one word at a time
        input      :   B*Maxl
        encoder_out   :   B*enc_imgsize*enc_imgsize*encoder_dim
        lenghts    :   B
        """
        # print("-"*10 + "Decoder forward" + "-"*10)
        batch_size = input.size(0)
        # print("[ Batch Size ] : " + str(batch_size))
        encoder_dim = encoder_out.size(3)
        Max_len = input.size(1)
        # print("[ Max_len ] : " + str(Max_len))
        # print("[ Encoder dimension ] : " + str(encoder_dim))
        
        embeddings =self.encoder(input) # B*Max_len*embed_dim
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # B*num_pixels*encoder_dim
        num_pixels = encoder_out.size(1)

        last_hidden, last_cell = self.init_hidden_state(encoder_out) #B*hidden_dim
        # print("[ Embedding Size ] : " + str(embeddings.size()))
        # print("[ Encoder_out Size] : " + str(encoder_out.size()))
        # print("[ Hidden Size, Cell size] : " + str(last_hidden.size()) + ", " + str(last_cell.size()))

        final_out = t.zeros(batch_size, Max_len, self.ntokens).to(device)
        alphas = t.zeros(batch_size, Max_len, num_pixels).to(device)
        for i in range(Max_len):
            batch_size_t = sum([l > i for l in lengths])
            attention_weighted_input, alpha = self.att(encoder_out[:batch_size_t], last_hidden[:batch_size_t]) # B*encoder_dim
            
            gate = self.sigmoid(self.f_beta(last_hidden[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_input = gate * attention_weighted_input
            # print("[ attention_weighted_input ] at " + str(i) + " : " + str(attention_weighted_input.size()))
            last_hidden, last_cell = self.lstm_cell(t.cat([embeddings[:batch_size_t, i, :], attention_weighted_input], dim=1), 
                                                            (last_hidden[:batch_size_t], last_cell[:batch_size_t]))  # B*hidden_dim


                                                # (torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), (h[:batch_size_t], c[:batch_size_t])) 
            out = self.decoder(self.dropout(last_hidden)) # 1*B*ntokens
            final_out[:batch_size_t,i,:] = out
            alphas[:batch_size_t,i,:] = alpha
        final_out = final_out.view(-1,self.ntokens)
        return final_out, alphas                                                                        

    def sample(self, encoder_out, beam_size):
        # print("-"*8 + " Sample forward " + "-"*8)
        k = beam_size
        encoder_dim = encoder_out.size(3)
        # print("[ Encoder dim ] :- " + str(encoder_dim))

        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) 
        h, c = self.init_hidden_state(encoder_out)
        # print("[ Sample h,c ] :- " + str(h.size()) + str(c.size()))

        k_prev_words = t.LongTensor([[self.word2idx['#START#']]]*k).to(device) 
        seqs = k_prev_words
        

        top_k_scores = t.zeros(k, 1).to(device)

        complete_seqs = [] # list of completed sequences 
        complete_seqs_scores = []
        complete_seq_alphas = []

        cnt = 0
        while True:
            # print("cnt - " + str(cnt))
            # inp_words = ""
            # for i in k_prev_words:
            #     inp_words = inp_words + self.idx2word[i] + " "
            # print(inp_words)
            embeddings = self.encoder(k_prev_words).squeeze(1) # s*embed_size
            # print("[ Embeddings ] :- " + str(embeddings.size()))

            attention_weighted_input, alpha = self.att(encoder_out, h) # s*encoder_dim
            # print("[ attention_weighted input ] :- " + str(attention_weighted_input.size()))

            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            attention_weighted_input = gate * attention_weighted_input

            h,c = self.lstm_cell(t.cat([embeddings, attention_weighted_input], dim = 1), (h,c))

            scores = self.decoder(h)
            scores = F.log_softmax(scores, dim = 1)
            scores = top_k_scores.expand_as(scores) + scores

            if cnt == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  
                k_prev_inds = top_k_words / self.ntokens
                for i, x in enumerate(k_prev_inds):
                    k_prev_inds[i] = i
                # seqs_alphas = alpha.unsqueeze(1)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
                k_prev_inds = top_k_words / self.ntokens
            next_k_inds = top_k_words % self.ntokens

            # print(k_prev_inds)
            # words = ""
            # for i in next_k_inds:
            #     words = words + self.idx2word[i] + " "
            # print(" words - " + words)
            seqs = t.cat([seqs[k_prev_inds], next_k_inds.unsqueeze(1)], dim = 1)
            # if(cnt != 0):
                # seqs_alphas = t.cat([seqs_alphas[k_prev_inds], alpha.unsqueeze(1)], dim = 1)
            # print("All seqs")
            # for i, lis in enumerate(seqs):
            #     cap = ""
            #     for j in lis:
            #         cap = cap + self.idx2word[j] + " "
            #     print(str(i) + " - " + cap)

            incomplete_inds = [ind for ind, word_ind in enumerate(next_k_inds) if self.idx2word[word_ind] != '.']
            complete_inds = [ind for ind, word_ind in enumerate(next_k_inds) if self.idx2word[word_ind] == '.']

            if (len(complete_inds) > 0):
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
                # complete_seq_alphas.extend(seqs_alphas[complete_inds].tolist())

            k -= len(complete_inds)
            if k == 0:
                break
            
            encoder_out = encoder_out[k_prev_inds[incomplete_inds]]
            k_prev_words = next_k_inds[incomplete_inds].unsqueeze(1)
            h = h[k_prev_inds[incomplete_inds]]
            c = c[k_prev_inds[incomplete_inds]]
            seqs = seqs[incomplete_inds]
            # seqs_alphas = seqs_alphas[incomplete_inds]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            # print(seqs_alphas.size())
            if cnt > 50:
                break   
            cnt = cnt + 1
            # print(cnt)
        # x = int(math.sqrt(num_pixels))
        # y = x
        # fig = plt.figure(figsize = (8,8))
        if (len(complete_seqs_scores) > 0):
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            out_cap = ""
            # rows = 2
            # cols = int((len(complete_seqs[i]))/2)
            for j, ind in enumerate(complete_seqs[i]):
                # print(self.idx2word[i])
                out_cap = out_cap + self.idx2word[ind] + " "
                # if(j < len(complete_seqs[i]) - 1):
                #     wei = np.reshape(complete_seq_alphas[i][j], [x,y])
                #     fig.add_subplot(rows, cols, j + 1)
                #     plt.imshow(wei, cmap = "gray")
                #     plt.title(str(j) + "th word")
            print(out_cap)
            print("All captions")
            for i, lis in enumerate(complete_seqs):
                cap = ""
                for j,ind in enumerate(lis):
                    cap = cap + self.idx2word[ind] + " "
                    # wei = complete_seq_alphas[i,j].view(x,y)
                print(cap + " - " + str(complete_seqs_scores[i]))
            # plt.show()
        else:
            print("No completed sentences!!!")
    # def sample(self,features):
    #     # """Generate captions for given image features using greedy search."""
    #     captions = []
    #     inputs = features.unsqueeze(1)
    #     for i in range(self.max_seg_length):
    #         hiddens, states = self.lstm(inputs)          # hiddens: (batch_size, 1, hidden_size)
    #         outputs = self.decoder(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
    #         _, predicted = outputs.max(1)
    #         # print(predicted)                        # predicted: (batch_size)
    #         # for ind in predicted.tolist():
    #             # print(self.idx2word[ind])
    #         captions.append([self.idx2word[ind] for ind in predicted.tolist()])
    #         inputs = self.encoder(predicted)                       # inputs: (batch_size, embed_size)
    #         inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
    #     # sampled_ids = t.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        
    #     return list(zip(*captions))
    


