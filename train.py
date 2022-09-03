import sys
import argparse
import model_dispatch
import warnings

warnings.filterwarnings("ignore")

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_datapath', type=str, required=True)
    parser.add_argument('-model', type=str, default='resnet18')
    parser.add_argument('-pretrained', default='True')
    parser.add_argument('-aug', default='False')
    return parser

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    train_datapath = namespace.train_datapath
    model = namespace.model
    pretrained = namespace.pretrained
    aug = namespace.aug

    if aug == 'False':
        net = model_dispatch.Model(train_datapath, model, pretrained)
    elif aug == 'True' and pretrained == 'True':
        net = model_dispatch.ModelAugmentation(train_datapath, model, pretrained)
    elif aug == 'True' and pretrained == 'False':
        print('Augmentation only for pretrained models!')
        sys.exit(0)
    else:
        print('Something went wrong!')
        sys.exit(0)

    net.model_choice()

    print("For checkpoints check folder 'checkpoints/'")



