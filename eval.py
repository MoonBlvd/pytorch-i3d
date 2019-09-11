import sys
sys.path.append('..')
from evaluation.eval_detection import ANETdetection

evaluator = ANETdetection(ground_truth_filename='A3D_i3d_label.json',
                          prediction_filename='tmp',
                          subset='val',
                          check_status=False)