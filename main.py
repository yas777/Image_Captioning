import torch as t
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
import math
import time
import data
import model
import argparse
import os
from random import shuffle
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--embed_size', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--emb_ft', type=bool, default=True,
                    help='size of word embeddings')                
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--attention_dim', type=int, default=512,
                    help='dimension in attenion layer')                    
parser.add_argument('--enc_lr', type=int, default=0.0001,
                    help='learning rate')
parser.add_argument('--dec_lr', type=int, default=0.001,
                    help='learning rate')                    
parser.add_argument('--bs', type=int, default=16,
                    help='batch_size')  
parser.add_argument('--load', type=str, default="",
                    help='loading the model')                                       
parser.add_argument('--save', type=str, default='../4/model',
                    help='path to save the final model')
parser.add_argument('--eval', type=bool, default=False,
                    help='whether to load the saved model')  
parser.add_argument('--cap_data', type=str, default='Datasets/flickr30k/',
                    help='location of the data corpus')     
parser.add_argument('--img_data', type=str, default='Datasets/flickr30k-images/',
                    help='location of the data corpus')
parser.add_argument('--imh', type=int, default=256,
                    help='location of the data corpus')
parser.add_argument('--imw', type=int, default=256,
                    help='location of the data corpus')                    
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')              
args = parser.parse_args()

device = t.device("cuda" if t.cuda.is_available() else "cpu")

path = './'
save_path = args.save
captions_path = os.path.join(path, args.cap_data)
img_dir = os.path.join(path, args.img_data)

corpus = data.Corpus(img_dir, captions_path)
word2idx = corpus.word2idx
# print(word2idx)
idx2word = corpus.idx2word
training_data = corpus.train_data
# print(training_data[1].size())
validation_data = corpus.val_data
ntokens = len(word2idx)

epochs = args.epochs
encoder_lr = args.enc_lr
decoder_lr = args.dec_lr
alpha_c = 1
batch_size = args.bs

imgh = args.imh
imgw = args.imw

embed_dim = args.embed_size
hidden_dim = args.nhid
attention_dim =args.attention_dim

transform = transforms.Compose([transforms.Resize((imgh, imgw)), 
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5),
														 (0.5, 0.5, 0.5))
									])
fine_tune_encoder = False
encoder = model.EncoderCNN().to(device)
encoder.fine_tune(fine_tune_encoder)
decoder = model.DecoderRNN(ntokens, embed_dim, hidden_dim, idx2word, word2idx).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

decoder_optimizer = t.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
encoder_optimizer = t.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return t.tensor(idxs, dtype=t.long, device = device)

def batchify(data, bs):
    shuffle(data)

    # print(data)
    n = len(data)
    num_batches = math.ceil(n/bs)
    bs_data = [0 for i in range(num_batches)]
    for i in range(num_batches):
        dat = data[i*bs:(i+1)*bs]
        dat.sort(key=lambda x : len(x[2]), reverse = True)
        imgs, caps, tars = zip(*dat)

        bs = len(caps) # B
        images = t.zeros(bs, 3, imgh, imgw)
        for j, img in enumerate(imgs):
            images[j] = transform(img)

        lengths = [len(cap) for cap in caps]
        captions = t.zeros(bs, max(lengths)).long() # B*Max_len
        targets = t.zeros(bs, max(lengths)).long() # B*Max_len
        for j, cap in enumerate(caps):
            end = lengths[j]
            captions[j,:end] = t.tensor(cap).long()
            targets[j, :(end)] = t.tensor(tars[j]).long()
        bs_data[i] = (images, captions, targets, lengths)
    return bs_data 

def evaluate(data_source):
    total_loss = 0
    encoder.eval()
    decoder.eval()
    i = 0
    with t.no_grad():
        for images, captions, targets, lengths in data_source:
            # print("val batch - " + str(i) + " - " + str(t.cuda.memory_allocated()))
            images = images.to(device)
            captions = captions.to(device)
            targets = targets.to(device)
            # targets = pack_padded_sequence(targets, lengths, batch_first = True)[0]

            # print(targets)
            features = encoder(images)
            outputs, alphas = decoder(captions, features, lengths)
            loss = loss_fn(outputs, targets.view(-1))
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            i = i + 1
            total_loss =total_loss+  loss.detach().cpu().numpy()
            # print(total_loss)
        # total_loss = total_loss/len(data_source)
    return total_loss/len(data_source)

def train(data):
    tl = 0
    i = 0
    encoder.train()
    decoder.train()
    for images, captions, targets, lengths in data:
        # print("batch nmber - " + str(i) + " - " + str(t.cuda.memory_allocated()))
        images = images.to(device)
        captions = captions.to(device)
        targets = targets.to(device)
        # targets = pack_padded_sequence(targets, lengths, batch_first = True)[0]

        # print(images.type())
        features = encoder(images)
        outputs, alphas = decoder(captions, features, lengths)

        # print("Output Size :- " + str(outputs.size()) + ", targets Size :- " + str(targets.size()))
        loss = loss_fn(outputs, targets.view(-1))
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        # print(t.cuda.memory_allocated(device))
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        tl += loss.detach().cpu().numpy()
        i = i + 1
    print("Training completed")
    return tl/len(data)

best_val_loss = None
cnt = 0
train_loss = 0
val_loss = 0
epoch = 0
if (args.load != ""):
    checkpoint = t.load(args.load)
    epoch = checkpoint['epoch'] + 1
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder_lr = checkpoint['encoder_lr']
    decoder_lr = checkpoint['decoder_lr']
    cnt = checkpoint['cnt']
    encoder_optimizer = checkpoint['encoder_optimizer']
    decoder_optimizer = checkpoint["decoder_optimizer"]
    for g in decoder_optimizer.param_groups:
        g['lr'] = decoder_lr
    if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    print("model loaded")

test_img1 = transform(Image.open(os.path.join(img_dir, "114715590.jpg")))
test_img2 = transform(Image.open(os.path.join(img_dir, "10951809.jpg")))
test_img3 = transform(Image.open(os.path.join(img_dir, "1131800850.jpg")))
test_img4 = transform(Image.open(os.path.join(img_dir, "1144874057.jpg")))
test_img5 = transform(Image.open(os.path.join(img_dir, "1051205546.jpg")))
tst_img1  = t.zeros(1,3,imgh,imgw)
tst_img2  = t.zeros(1,3,imgh,imgw)
tst_img3  = t.zeros(1,3,imgh,imgw)
tst_img4  = t.zeros(1,3,imgh,imgw)
tst_img5  = t.zeros(1,3,imgh,imgw)
tst_img1[0,:,:,:] = test_img1
tst_img2[0,:,:,:] = test_img2
tst_img3[0,:,:,:] = test_img3
tst_img4[0,:,:,:] = test_img4
tst_img5[0,:,:,:] = test_img5
# print(test_img.unsqueeze(0).size())


if (not args.eval):
    tr_data = batchify(training_data, batch_size)
    val_data = batchify(validation_data, batch_size)
    for epoch in range(epoch, epochs):
        epoch_start_time = time.time()
        
        train_loss = train(tr_data)
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | training loss {:5.5f} | valid loss {:5.5f} | '
        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), train_loss,
        val_loss, math.exp(val_loss)))
        print("generating caption1")
        encoder.eval()
        decoder.eval()
        with t.no_grad():
            test_img = tst_img1.to(device)
            test_features = encoder(test_img)
            decoder.sample(test_features, 10)
            print("generating caption2")
            test_img = tst_img2.to(device)
            test_features = encoder(test_img)
            decoder.sample(test_features, 10)
            print("generating caption3")
            test_img = tst_img3.to(device)
            test_features = encoder(test_img)
            decoder.sample(test_features, 10)
            print("generating caption4")
            test_img = tst_img4.to(device)
            test_features = encoder(test_img)
            decoder.sample(test_features, 10)
            print("generating caption5")
            test_img = tst_img5.to(device)
            test_features = encoder(test_img)
            decoder.sample(test_features, 10)
        # print(cap)
        print('-' * 89)
        with open(save_path, 'wb') as f:
            t.save({
                'epoch' : epoch,
                'encoder' : encoder,
                'decoder' : decoder,
                'encoder_lr' : encoder_lr,
                'decoder_lr' : decoder_lr,
                'cnt' : cnt,
                'encoder_optimizer' : encoder_optimizer,
                'decoder_optimizer' : decoder_optimizer
            },path+args.save+"_50_1_"+str(epoch) + ".pt")
            print("Model saved")
        # if not best_val_loss or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     cnt = 0
        # else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
        cnt = cnt+1
        if (cnt == 8 and epoch > 39) or (cnt == 5 and epoch < 39):
            for g in decoder_optimizer.param_groups:
                g['lr']*=0.5
                decoder_lr/=2.0
            print('Decreasing learning rate')
            cnt = 0

encoder.eval()
decoder.eval()
with t.no_grad():
    test_img = tst_img1.to(device)
    test_features = encoder(test_img)
    decoder.sample(test_features, 10)
    print("generating caption2")
    test_img = tst_img2.to(device)
    test_features = encoder(test_img)
    decoder.sample(test_features, 10)
    print("generating caption3")
    test_img = tst_img3.to(device)
    test_features = encoder(test_img)
    decoder.sample(test_features, 10)
    print("generating caption4")
    test_img = tst_img4.to(device)
    test_features = encoder(test_img)
    decoder.sample(test_features, 10)
    print("generating caption5")
    test_img = tst_img5.to(device)
    test_features = encoder(test_img)
    decoder.sample(test_features, 10)
        # print(cap)




