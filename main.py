from nq_flags import DEFAULT_FLAGS as FLAGS
from nq_flags import del_all_flags
from nq_dataset_utils import *
import transformers
import os
import sys
import json
import tensorflow as tf
import numpy as np
import absl
import datetime
from model import *
from learning_scheduler import *
from tensorflow.keras.optimizers import Adam
from adamw_optimizer import AdamW

flags = absl.flags
del_all_flags(flags.FLAGS)
BASE_DIR = '.'

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

vocab_file = os.path.join(BASE_DIR, "vocab-nq.txt")

flags.DEFINE_string("vocab_file", vocab_file,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer(
    "max_seq_length_for_training", 512,
    "The maximum total input sequence length after WordPiece tokenization for training examples. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_float(
    "include_unknowns_for_training", 0.02,
    "If positive, for converting training dataset, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

TRAIN_TF_RECORD = os.path.join(BASE_DIR, "nq_train.tfrecord")
    
flags.DEFINE_string("train_tf_record", TRAIN_TF_RECORD,
                    "Precomputed tf records for training dataset.")

flags.DEFINE_string("valid_tf_record", os.path.join(BASE_DIR, "nq_valid.tfrecord"),
                    "Precomputed tf records for validation dataset.")

flags.DEFINE_string("valid_small_tf_record", os.path.join(BASE_DIR, "nq_valid_small.tfrecord"),
                    "Precomputed tf records for a smaller validation dataset.")

flags.DEFINE_string("valid_tf_record_with_labels", "nq_valid_with_labels.tfrecord",
                    "Precomputed tf records for validation dataset with labels.")

flags.DEFINE_string("valid_small_tf_record_with_labels", "nq_valid_small_with_labels.tfrecord",
                    "Precomputed tf records for a smaller validation dataset with labels.")

# This file should be generated when the kernel is running using the provided test dataset!
flags.DEFINE_string("test_tf_record", "nq_test.tfrecord",
                    "Precomputed tf records for test dataset.")

flags.DEFINE_bool("do_train", False, "Whether to run training dataset.")

flags.DEFINE_bool("do_valid", False, "Whether to run validation dataset.")

flags.DEFINE_bool("smaller_valid_dataset", True, "Whether to use the smaller validation dataset")

flags.DEFINE_bool("do_predict", True, "Whether to run test dataset.")

flags.DEFINE_string(
    "validation_prediction_output_file", "validatioin_predictions.json",
    "Where to print predictions for validation dataset in NQ prediction format, to be passed to natural_questions.nq_eval.")

flags.DEFINE_string(
    "validation_small_prediction_output_file", "validatioin_small_predictions.json",
    "Where to print predictions for validation dataset in NQ prediction format, to be passed to natural_questions.nq_eval.")

flags.DEFINE_string(
    "prediction_output_file", "predictions.json",
    "Where to print predictions for test dataset in NQ prediction format, to be passed to natural_questions.nq_eval.")

flags.DEFINE_string(
    "input_checkpoint_dir", os.path.join(BASE_DIR, "checkpoints"),
    "The root directory that contains checkpoints to be loaded of all trained models.")

flags.DEFINE_string(
    "output_checkpoint_dir", "checkpoints",
    "The output directory where the model checkpoints will be written to.")

# If you want to use other Hugging Face's models, change this to `BASE_DIR` and put the downloaded models at the right place.
flags.DEFINE_string("model_dir", BASE_DIR, "Root dir of all Hugging Face's models")

flags.DEFINE_string("model_name", "distilbert-base-uncased-distilled-squad", "Name of Hugging Face's model to use.")
# flags.DEFINE_string("model_name", "bert-base-uncased", "Name of Hugging Face's model to use.")
# flags.DEFINE_string("model_name", "bert-large-uncased-whole-word-masking-finetuned-squad", "Name of Hugging Face's model to use.")

flags.DEFINE_integer("epochs", 1, "Total epochs for training.")

flags.DEFINE_integer("train_batch_size", 10, "Batch size for training.")

flags.DEFINE_integer("shuffle_buffer_size", 100000, "Shuffle buffer size for training.")

flags.DEFINE_integer("batch_accumulation_size", 50, "Number of batches to accumulate gradient before applying optimization.")

flags.DEFINE_float("init_learning_rate", 5e-5, "The initial learning rate for AdamW optimizer.")

flags.DEFINE_bool("cyclic_learning_rate", True, "If to use cyclic learning rate.")

flags.DEFINE_float("init_weight_decay_rate", 0.01, "The initial weight decay rate for AdamW optimizer.")

flags.DEFINE_integer("num_warmup_steps", 0, "Number of training steps to perform linear learning rate warmup.")

flags.DEFINE_integer("num_train_examples", None, "Number of precomputed training steps in 1 epoch.")

flags.DEFINE_integer("predict_batch_size", 25, "Batch size for predictions.")

# ----------------------------------------------------------------------------------------
flags.DEFINE_integer(
    "n_best_size", 10,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_string(
    "validation_predict_file", os.path.join(BASE_DIR, "simplified-nq-dev.jsonl"),
    "")

flags.DEFINE_string(
    "validation_predict_file_small", os.path.join(BASE_DIR, "simplified-nq-dev-small.jsonl"),
    "")

# ----------------------------------------------------------------------------------------
## Special flags - do not change

flags.DEFINE_string(
    "predict_file", os.path.join(BASE_DIR, "simplified-nq-test.jsonl"),
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz") 
    
flags.DEFINE_boolean("logtostderr", True, "Logs to stderr")
flags.DEFINE_boolean("undefok", True, "it's okay to be undefined")
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('HistoryManager.hist_file', '', 'kernel')

# Make the default flags as parsed flags
FLAGS.mark_as_parsed()

def get_dataset(tf_record_file, seq_length, batch_size=1, shuffle_buffer_size=0, is_training=False):

    if is_training:
        features = {
            "unique_ids": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "start_positions": tf.io.FixedLenFeature([], tf.int64),
            "end_positions": tf.io.FixedLenFeature([], tf.int64),
            "answer_types": tf.io.FixedLenFeature([], tf.int64)
        }
    else:
        features = {
            "unique_ids": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64)
        }        

    # Taken from the TensorFlow models repository: https://github.com/tensorflow/models/blob/befbe0f9fe02d6bc1efb1c462689d069dae23af1/official/nlp/bert/input_pipeline.py#L24
    def decode_record(record, features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        return example

    def select_data_from_record(record):
        
        x = {
            'unique_ids': record['unique_ids'],
            'input_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'segment_ids': record['segment_ids']
        }

        if is_training:
            y = {
                'start_positions': record['start_positions'],
                'end_positions': record['end_positions'],
                'answer_types': record['answer_types']
            }

            return (x, y)
        
        return x

    dataset = tf.data.TFRecordDataset(tf_record_file)
    
    dataset = dataset.map(lambda record: decode_record(record, features))
    dataset = dataset.map(select_data_from_record)
    
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    
    dataset = dataset.batch(batch_size)
    
    return dataset

if FLAGS.num_train_examples is None:
    FLAGS.num_train_examples = 494670

def get_metrics(name):

    loss = tf.keras.metrics.Mean(name=f'{name}_loss')
    loss_start_pos = tf.keras.metrics.Mean(name=f'{name}_loss_start_pos')
    loss_end_pos = tf.keras.metrics.Mean(name=f'{name}_loss_end_pos')
    loss_ans_type = tf.keras.metrics.Mean(name=f'{name}_loss_ans_type')
    
    acc = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc')
    acc_start_pos = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc_start_pos')
    acc_end_pos = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc_end_pos')
    acc_ans_type = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc_ans_type')
    
    return loss, loss_start_pos, loss_end_pos, loss_ans_type, acc, acc_start_pos, acc_end_pos, acc_ans_type

def loss_function(nq_labels, nq_logits):
    
    (start_pos_labels, end_pos_labels, answer_type_labels) = nq_labels
    (start_pos_logits, end_pos_logits, answer_type_logits) = nq_logits
    
    loss_start_pos = loss_object(start_pos_labels, start_pos_logits)
    loss_end_pos = loss_object(end_pos_labels, end_pos_logits)
    loss_ans_type = loss_object(answer_type_labels, answer_type_logits)
    
    loss_start_pos = tf.math.reduce_sum(loss_start_pos)
    loss_end_pos = tf.math.reduce_sum(loss_end_pos)
    loss_ans_type = tf.math.reduce_sum(loss_ans_type)
    
    loss = (loss_start_pos + loss_end_pos + loss_ans_type) / 3.0
    
    return loss, loss_start_pos, loss_end_pos, loss_ans_type

def get_loss_and_gradients(input_ids, input_masks, segment_ids, start_pos_labels, end_pos_labels, answer_type_labels):
    
    nq_inputs = (input_ids, input_masks, segment_ids)
    nq_labels = (start_pos_labels, end_pos_labels, answer_type_labels)

    with tf.GradientTape() as tape:

        nq_logits = bert_nq(nq_inputs, training=True)
        loss, loss_start_pos, loss_end_pos, loss_ans_type = loss_function(nq_labels, nq_logits)
                
    gradients = tape.gradient(loss, bert_nq.trainable_variables)        
        
    (start_pos_logits, end_pos_logits, answer_type_logits) = nq_logits
        
    train_acc.update_state(start_pos_labels, start_pos_logits)
    train_acc.update_state(end_pos_labels, end_pos_logits)
    train_acc.update_state(answer_type_labels, answer_type_logits)

    train_acc_start_pos.update_state(start_pos_labels, start_pos_logits)
    train_acc_end_pos.update_state(end_pos_labels, end_pos_logits)
    train_acc_ans_type.update_state(answer_type_labels, answer_type_logits)
    
    return loss, gradients, loss_start_pos, loss_end_pos, loss_ans_type

input_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
]
train_loss, train_loss_start_pos, train_loss_end_pos, train_loss_ans_type, train_acc, train_acc_start_pos, train_acc_end_pos, train_acc_ans_type = get_metrics("train")
valid_loss, valid_loss_start_pos, valid_loss_end_pos, valid_loss_ans_type, valid_acc, valid_acc_start_pos, valid_acc_end_pos, valid_acc_ans_type = get_metrics("valid")


@tf.function(input_signature=input_signature)
def train_step_simple(input_ids, input_masks, segment_ids, start_pos_labels, end_pos_labels, answer_type_labels):

    nb_examples = tf.math.reduce_sum(tf.cast(tf.math.not_equal(start_pos_labels, -2), tf.int32))
    
    loss, gradients, loss_start_pos, loss_end_pos, loss_ans_type = get_loss_and_gradients(input_ids, input_masks, segment_ids, start_pos_labels, end_pos_labels, answer_type_labels)
    
    average_loss = tf.math.divide(loss, tf.cast(nb_examples, tf.float32))
    
    # For this simple training step, it's better to use tf.math.reduce_mean() in loss_function() instead of tf.math.reduce_sum(), and not using the following line
    # to average gradients manually.
    
    # Using this line causing `UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape.`.
    average_gradients = [tf.divide(x, tf.cast(nb_examples, tf.float32)) for x in gradients]
    
    optimizer.apply_gradients(zip(gradients, bert_nq.trainable_variables))

    average_loss_start_pos = tf.math.divide(loss_start_pos, tf.cast(nb_examples, tf.float32))
    average_loss_end_pos = tf.math.divide(loss_end_pos, tf.cast(nb_examples, tf.float32))
    average_loss_ans_type = tf.math.divide(loss_ans_type, tf.cast(nb_examples, tf.float32))
    
    train_loss(average_loss)
    train_loss_start_pos(average_loss_start_pos)
    train_loss_end_pos(average_loss_end_pos)
    train_loss_ans_type(average_loss_ans_type)

bert_tokenizer, bert_nq = get_pretrained_model(FLAGS.model_dir, FLAGS.model_name)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

num_train_steps = int(FLAGS.epochs * FLAGS.num_train_examples / FLAGS.train_batch_size / FLAGS.batch_accumulation_size)

learning_rate = CustomSchedule(
    initial_learning_rate=FLAGS.init_learning_rate,
    decay_steps=num_train_steps,
    end_learning_rate=FLAGS.init_learning_rate,
    power=1.0,
    cycle=FLAGS.cyclic_learning_rate,    
    num_warmup_steps=FLAGS.num_warmup_steps
)

decay_var_list = []

for i in range(len(bert_nq.trainable_variables)):
    name = bert_nq.trainable_variables[i].name
    if any(x in name for x in ["LayerNorm", "layer_norm", "bias"]):
        decay_var_list.append(name)

# The hyperparameters are copied from AdamWeightDecayOptimizer in original bert code.
# (https://github.com/google-research/bert/blob/master/optimization.py#L25)
optimizer = AdamW(weight_decay=FLAGS.init_weight_decay_rate, learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay_var_list=decay_var_list)

checkpoint_path = os.path.join(FLAGS.input_checkpoint_dir, FLAGS.model_name)
ckpt = tf.train.Checkpoint(model=bert_nq)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10000)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print (f'Latest BertNQ checkpoint restored -- Model trained for {last_epoch} epochs')
else:
    print('Checkpoint not found. Train BertNQ from scratch')
    last_epoch = 0
    
    
print(ckpt_manager._directory)
ckpt_manager._directory = os.path.join(FLAGS.output_checkpoint_dir, FLAGS.model_name)
ckpt_manager._checkpoint_prefix = os.path.join(ckpt_manager._directory, "ckpt")
print(ckpt_manager._directory)

from tensorflow.python.lib.io.file_io import recursive_create_dir
recursive_create_dir(ckpt_manager._directory)

train_step = train_step_simple
train_start_time = datetime.datetime.now()

epochs = FLAGS.epochs

for epoch in range(epochs):

    train_dataset = get_dataset(
        FLAGS.train_tf_record,
        FLAGS.max_seq_length_for_training,
        FLAGS.batch_accumulation_size * FLAGS.train_batch_size,
        FLAGS.shuffle_buffer_size,
        is_training=True
    )     
    
    train_loss.reset_states()
    train_loss_start_pos.reset_states()
    train_loss_end_pos.reset_states()
    train_loss_ans_type.reset_states()    
    
    train_acc.reset_states()
    train_acc_start_pos.reset_states()
    train_acc_end_pos.reset_states()
    train_acc_ans_type.reset_states()
    
    epoch_start_time = datetime.datetime.now()
    
    for (batch_idx, (features, targets)) in enumerate(train_dataset):        
        (input_ids, input_masks, segment_ids) = (features['input_ids'], features['input_mask'], features['segment_ids'])
        (start_pos_labels, end_pos_labels, answer_type_labels) = (targets['start_positions'], targets['end_positions'], targets['answer_types'])
    
        batch_start_time = datetime.datetime.now()
        
        train_step(input_ids, input_masks, segment_ids, start_pos_labels, end_pos_labels, answer_type_labels)

        batch_end_time = datetime.datetime.now()
        batch_elapsed_time = (batch_end_time - batch_start_time).total_seconds()
        
        if (batch_idx + 1) % 100 == 0:
            print('Epoch {} | Batch {} | Elapsed Time {}'.format(
                epoch + 1,
                batch_idx + 1,
                batch_elapsed_time
            ))
            print('Loss {:.6f} | Loss_S {:.6f} | Loss_E {:.6f} | Loss_T {:.6f}'.format(
                train_loss.result(),
                train_loss_start_pos.result(),
                train_loss_end_pos.result(),
                train_loss_ans_type.result()
            ))
            print(' Acc {:.6f} |  Acc_S {:.6f} |  Acc_E {:.6f} |  Acc_T {:.6f}'.format(
                train_acc.result(),
                train_acc_start_pos.result(),
                train_acc_end_pos.result(),
                train_acc_ans_type.result()
            ))
            print("-" * 100)
       
    epoch_end_time = datetime.datetime.now()
    epoch_elapsed_time = (epoch_end_time - epoch_start_time).total_seconds()
            
    if (epoch + 1) % 1 == 0:
        
        ckpt_save_path = ckpt_manager.save()
        print ('\nSaving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        
        print('\nEpoch {}'.format(epoch + 1))
        print('Loss {:.6f} | Loss_S {:.6f} | Loss_E {:.6f} | Loss_T {:.6f}'.format(
            train_loss.result(),
            train_loss_start_pos.result(),
            train_loss_end_pos.result(),
            train_loss_ans_type.result()
        ))
        print(' Acc {:.6f} |  Acc_S {:.6f} |  Acc_E {:.6f} |  Acc_T {:.6f}'.format(
            train_acc.result(),
            train_acc_start_pos.result(),
            train_acc_end_pos.result(),
            train_acc_ans_type.result()
        ))

    print('\nTime taken for 1 epoch: {} secs\n'.format(epoch_elapsed_time))
    print("-" * 80 + "\n")
