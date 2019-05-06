Image_Captioning  
  ![a man is drving a atv in the field](./Datasets/flickr30k-images/109823395.jpg)  
  Our project is to generate captions for a given image. We imlemented [show attend and tell](https://arxiv.org/pdf/1502.03044.pdf) paper.
Model:  
  Encoder:   
    The Encoder encodes the image and outputs a smaller iamge(14*14) with 2048 learned which acts as the summary representaion of input image   
    We used 52 layered Residual Network trained on the ImageNet classification task, already available in PyTorch.
  
  
