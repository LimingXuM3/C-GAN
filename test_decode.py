from steganoganphy import CGAN
from models import BasicCritic
from models import DenseDecoder
from models import BasicEncoder, DenseEncoder, ResidualEncoder

def main():
    steganogan = CGAN(3, BasicEncoder, DenseDecoder, BasicCritic, hidden_size=32, cuda=True, verbose=True)
    steganogan = steganogan.load('weights', cuda=False)
    steganogan.decode('steg.jpg')

if __name__ == '__main__':
    main()
