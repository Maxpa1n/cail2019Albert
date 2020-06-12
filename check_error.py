import json

from evaluate import CJRCEvaluator

dev_file = 'data/big_dev.json'
predictions_file = 'output_dir/predictions.json'

if __name__ == '__main__':
    with open(dev_file, 'r', encoding='utf8') as f:
        dev_data = json.load(f)
    with open(predictions_file, 'r', encoding='utf8') as f:
        all_predictions = json.load(f)

    evaluator = CJRCEvaluator(dev_file)
    res = evaluator.model_performance(all_predictions)
