import os
import json
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

import torch
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from pytorch_pretrain_bert.modeling import PROP, BertConfig
from pytorch_pretrain_bert.tokenization import BertTokenizer
from pytorch_pretrain_bert.optimization import BertAdam, warmup_linear

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids label lm_label_ids ")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def convert_example_to_features(example, max_seq_length):
    label = example["label"]
    input_ids = example["input_ids"]
    segment_ids = example["segment_ids"]
    masked_label_ids = example["masked_label_ids"]
    masked_lm_positions = example["masked_lm_positions"]

    # The preprocessed data should be already truncated
    assert len(input_ids) == len(segment_ids) <= max_seq_length

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.int)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.int)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             label=label
                             )
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, num_data_epochs, temp_dir='./', mode='train'):
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        if mode == 'train':
            # Samples for one epoch should not larger than 26000000
            if num_samples > 26000000:
                num_samples = 26000000
        else:
            num_samples = 1000 # NOT USE
        
        self.temp_dir = None
        self.working_dir = None
        seq_len = metrics['max_seq_len']
        self.temp_dir = TemporaryDirectory(dir=temp_dir)
        self.working_dir = Path(self.temp_dir.name)
        input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
        input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
        segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
        labels = np.memmap(filename=self.working_dir/'labels.memmap',
                                shape=(num_samples), mode='w+', dtype=np.bool)
        lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
        lm_label_ids[:] = -1

        logging.info(f"Loading {mode} examples for epoch {epoch}")
        with data_file.open() as f:
            instance_index = 0
            for i, line in enumerate(tqdm(f, total=num_samples, desc=f"{mode} examples")):
                if i+1 > num_samples:
                    break
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, seq_len)
                input_ids[instance_index] = features.input_ids
                segment_ids[instance_index] = features.segment_ids
                input_masks[instance_index] = features.input_mask
                labels[instance_index] = features.label
                lm_label_ids[i] = features.lm_label_ids
                instance_index += 1
        logging.info('Real num samples:{}'.format(instance_index))
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.labels = labels
        self.lm_label_ids = lm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(int(self.labels[item])),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                )

class RandomPairSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, negtive=1):
        self.data_source = data_source
        self.negtive = negtive
        if (len(self.data_source)%(self.negtive+1)) !=0:
            raise ValueError('data length {} % {} !=0, can not pair data!'.format(len(self.data_source), self.negtive+1))
    
    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        indices = torch.arange(len(self.data_source))
        paired_indices = indices.unfold(0, self.negtive+1, self.negtive+1)
        paired_indices = torch.stack([paired_indices[i] for i in range(len(paired_indices))])
        paired_indices = paired_indices[torch.randperm(len(paired_indices))]
        indices = paired_indices.view(-1)
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.data_source)


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--temp_dir", type=str, default='./')
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--negtive_num",
                        type=int,
                        default=1,
                        help="Nums of negtive exmaples for one positive example.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--save_checkpoints_steps",
                        default=10000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            # Samples for one epoch should not larger than 26000000
            metrics['num_training_examples'] = metrics['num_training_examples'] if metrics['num_training_examples'] < 26000000 else 26000000
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = PROP.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        # try:
        #     from apex.parallel import DistributedDataParallel as DDP
        # except ImportError:
        #     raise ImportError(
        #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        # model = DDP(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        epoch_train_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data,
                                            num_data_epochs=num_data_epochs, temp_dir=args.temp_dir)
        epoch_eval_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data,
                                            num_data_epochs=num_data_epochs, temp_dir=args.temp_dir, mode='eval')
        if args.local_rank == -1:
            train_sampler = RandomPairSampler(epoch_train_dataset, args.negtive_num)
            eval_sampler = SequentialSampler(epoch_eval_dataset)
        else:
            # Not supported
            train_sampler = DistributedSampler(epoch_train_dataset)
            eval_sampler = DistributedSampler(epoch_eval_dataset)
        train_dataloader = DataLoader(epoch_train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        eval_dataloader = DataLoader(epoch_eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        tr_loss = 0
        nb_tr_steps = 0
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {total_train_examples}")
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label, lm_label_ids = batch
                
                loss = model(input_ids, segment_ids, input_mask, lm_label_ids, label)
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_steps += 1
                pbar.update(1)
                
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                writer.add_scalar('train/loss', round(mean_loss,4), global_step)
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
    
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
    
                    if global_step % args.save_checkpoints_steps == 0:
                        with torch.no_grad():
                            # Save a ckpt
                            logging.info("** ** * Saving model ** ** * ")
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            output_model_file = args.output_dir / "pytorch_model_{}.bin".format(global_step)
                            torch.save(model_to_save.state_dict(), str(output_model_file))
    
    # Save the last model
    logging.info("** ** * Saving model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = args.output_dir / "pytorch_model_last.bin"
    torch.save(model_to_save.state_dict(), str(output_model_file))
    writer.close()

if __name__ == '__main__':
    main()
