
import argparse


class ArgumentParserModule:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train GraphGPT')
        self.parser.add_argument('--config', type=str, default='configs/default.yaml',
                                 help='Path to configuration file (default: config.yaml)')
        self.parser.add_argument('--gpu', nargs='+', type=int, default=None,
                                 help='ID(s) of the GPU(s) to use (default: CPU)')

    def parse_args(self):
        args = self.parser.parse_args()
        
        # Convert args.gpu to a list if it is an integer
        if isinstance(args.gpu, int):
            args.gpu = [args.gpu]
        return args