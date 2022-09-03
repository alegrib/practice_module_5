import sys
import argparse
import model_dispatch
import warnings

warnings.filterwarnings("ignore")

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_datapath', type=str, required=True)
    parser.add_argument('-model', type=str, default='VGG')
    parser.add_argument('-pretrained', default='True')
    return parser

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    test_datapath = namespace.test_datapath
    model = namespace.model
    pretrained = namespace.pretrained

    net = model_dispatch.ModelEval(test_datapath, model, pretrained)
    net.test_eval()



