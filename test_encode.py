from steganoganphy import CGAN
from models import BasicCritic
from models import DenseDecoder
from models import BasicEncoder, DenseEncoder, ResidualEncoder

def main():
    steganogan = CGAN(3, BasicEncoder, DenseDecoder, BasicCritic, hidden_size=32, cuda=True, verbose=True)
    steganogan = steganogan.load('weights', cuda=False)
    steganogan.encode('image.jpg', 'steg.jpg', 'This is a super secret message! This is a super secret message!')

if __name__ == '__main__':
    main()
