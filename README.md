*Image_Captioning*  
  ![a man is drving a atv in the field](./Datasets/flickr30k-images/109823395.jpg)  
  Our project is to generate captions for a given image. We imlemented [show attend and tell](https://arxiv.org/pdf/1502.03044.pdf) paper.
*Model*:  
  *Encoder*:   
    The Encoder encodes the image and outputs a smaller iamge(14*14) with 2048 learned which acts as the summary representaion of input  image   
    We used 52 layered Residual Network trained on the ImageNet classification task, already available in PyTorch.
  * Attention*:
    Attention network computes the weights corresponding to each encoded pixels which reflects the importance of that pixel in effecting the prediction of next word.
    The weights corresponding to each pixel at a time instant are calculated using hidden state of previous time instant and the encoder output. We used soft attention where the sum of weigths of pixels add up to one.
  *Decoder*:
    By looking at the encoded image with probabilities assigned to each pixels, last hidden it tries to predict the next word in the sequence.
    We used lstm since generating a sequence would need recurrent neural network.
Requirements:   
  * python 3.5+
  * pytorch 
  * Numpy
  * cuda
First time set up:  
      
  
  
  
  
