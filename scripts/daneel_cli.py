import argparse 
import yaml

from daneel.atmosphere.spectral_modeler import ForwardModelRunner, RetrievalModelRunner

parser = argparse.ArgumentParser(description='Daneel Computational Astrophysics Tool')
parser.add_argument('-i', help='Path to the parameters YAML file.', required=True)
parser.add_argument('-d', '--detection', choices=['svm', 'nn', 'cnn'], help='Detection algorithm to use.')
parser.add_argument('--dream', action='store_true', help='Flag to run the GAN "dream" task.')
parser.add_argument('-a', '--atmosphere', choices=['model', 'retrieve'], help='Atmosphere task to perform: "model" or "retrieve"')

args = parser.parse_args()

with open(args.i, 'r') as f:
    
    config = yaml.safe_load(f)
    
if args.atmosphere == 'model':
    atmosphere_config = config.get('atmosphere', {})

    try:
        runner = ForwardModelRunner(atmosphere_config)
        runner.run()
        print("Atmosphere Forward Model complete. Spectrum saved.")
    except NameError:
        print("ERROR: ForwardModelRunner class is not defined. Cannot run Task E.")
    
elif args.atmosphere == 'retrieve':
    atmosphere_config = config.get('atmosphere', {})

    runner = RetrievalModelRunner(atmosphere_config)
    runner.run()
    print("Atmosphere Retrieval Model complete. Results saved.")

elif args.detection:
    from daneel.detection.svm_detector import SVMDetector
    from daneel.detection.nn_detector import NNDetector
    from daneel.detection.cnn_detector import CNNDetector
    
    detection_config = config['detection']
    
    if args.detection == 'svm':
        svm_config = config['detection']['svm']
        detector = SVMDetector(
            dataset_path=svm_config['dataset_path'],
            kernel_name=svm_config['kernel'],
            degree=svm_config.get('degree')
        )
        detector.run_detection()
    elif args.detection == 'nn':
        nn_config = config['detection']['nn']

        detector = NNDetector(
            dataset_path=nn_config['dataset_path'],
            kernel_name=None
        )
        detector.run_detection()
    elif args.detection == 'cnn':
        cnn_config = config['detection']['cnn']
        detector = CNNDetector(
            dataset_path=cnn_config['dataset_path']
        )
        detector.run_detection()

elif args.dream:
    from daneel.dream.gan import GAN
    dream_cfg = config.get('dream', {})
    # pass the top-level params you need
    params = {
        'dataset_path': dream_cfg.get('dataset_path', config['detection']['cnn']['dataset_path']),
        'out_dir': dream_cfg.get('out_dir', './data'),
        'device': dream_cfg.get('device', 'cuda'),
        'nz': dream_cfg.get('nz', 100),
        'epochs': dream_cfg.get('epochs', 50),
        'batch_size': dream_cfg.get('batch_size', 64),
        'lr': dream_cfg.get('lr', 2e-4),
    }
    gan = GAN(params)
    # either train or directly dream depending on config
    if dream_cfg.get('train', True):
        gan.train()
    # optionally generate images
    if dream_cfg.get('dream', True):
        gan.dream(n_images=dream_cfg.get('n_images', 16), checkpoint=dream_cfg.get('checkpoint', None))