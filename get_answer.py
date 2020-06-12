import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert.tokenization import BertTokenizer

from CailExample import convert_examples_to_features, write_predictions_test, \
    SquadExample
from CailModelAlbert import CailModel
from run_albert_cail import RawResult

max_seq_length = 512
doc_stride = 128
max_query_length = 64
max_answer_length = 30
output_dir = 'checkpoint/'


def read_squad_examples_get(content, question):
    examples = []
    doc_tokens = []
    char_to_word_offset = []

    for c in content:
        doc_tokens.append(c)
        char_to_word_offset.append(len(doc_tokens) - 1)

    qas_id = "7777777"
    question_text = question
    start_position = None
    end_position = None
    orig_answer_text = None
    is_impossible = False
    is_yes = False
    is_no = False
    example = SquadExample(
        qas_id=qas_id,
        question_text=question_text,
        doc_tokens=doc_tokens,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        end_position=end_position,
        is_impossible=is_impossible,
        is_yes=is_yes,
        is_no=is_no)
    examples.append(example)
    return examples


def load_test_features_get(tokenizer, content, question):
    test_examples = read_squad_examples_get(content, question)

    test_features = convert_examples_to_features(
        examples=test_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler)

    return test_dataloader, test_examples, test_features


def do_reading(content, question, model, device):
    #
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
    test_dataloader, test_examples, test_features = load_test_features_get(tokenizer, content, question)
    model.to(device)
    model.eval()
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, \
            batch_unk_logits, batch_yes_logits, batch_no_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            unk_logits = batch_unk_logits[i].detach().cpu().tolist()
            yes_logits = batch_yes_logits[i].detach().cpu().tolist()
            no_logits = batch_no_logits[i].detach().cpu().tolist()
            test_feature = test_features[example_index.item()]
            unique_id = int(test_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         unk_logits=unk_logits,
                                         yes_logits=yes_logits,
                                         no_logits=no_logits))

    preds = write_predictions_test(test_examples, test_features, all_results, 1,
                                   max_answer_length, True, output_dir+"preds.json", True, True, True)

    return preds


if __name__ == '__main__':
    context = "根据当事人陈述和经审查确认的证据,本院认定事实如下:原告的被保险人王有文于2014年11月26日将其购买的一批水果交由被告苏x5负责承运,从广州江南市场运往新疆伊犁;承运车辆为宁D×××××(宁D×××××),同年11月30日,当车辆运行至连霍高速新疆盐湖收费站路段时,因司机刘x4操作不当,与前方行驶车辆发生追尾,导致所运水果受损事故发生后,原告根据保险合同向被保险人支付了121571.03元保险金,依法取得了代位求偿权原告没有提交证明所运输的水果在事发前已经告知过被告苏x5的证据;在原告提供的公估案件现场清点、查勘记录中,只有表一有被告苏x5的签字确认;原告当庭按照被告苏x5签字确认的现场清点查勘记录所记载的受损水果品种及公估报告中所对应的该些水果价格,计算得出受损金额为102890.85元"
    question = "向原告投保的人是谁？"

    device = 'cpu'
    model = CailModel.from_pretrained(output_dir)
    preds = do_reading(context, question, model, device)
    print(preds)
