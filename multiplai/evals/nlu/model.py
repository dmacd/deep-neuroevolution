
import numpy as np
import mxnet as mx
from io import open
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, Block
from mxnet import ndarray as F
from mxnet import ndarray as nd
from typing import List, Set, Tuple, Dict

from multiplai.evals.nlu import data
from multiplai.evals.nlu import embedding as emb

#
# class EncoderRNN(Block):
#     def __init__(self, word_embedding, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         with self.name_scope():
#             # self.embedding = word_embedding #nn.Embedding(input_size, hidden_size)
#             self.embedding = emb.layer_for_embedding(word_embedding)
#             # TODO: assert somehow that the output size is the same
#             # could use 'infer_shape' https://github.com/apache/incubator-mxnet/issues/1090
#
#             self.rnn = rnn.GRU(hidden_size,
#                                input_size=word_embedding.idx_to_vec.shape[1])
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).swapaxes(0, 1)
#         output, hidden = self.rnn(output, hidden)
#         return output, hidden
#
#     def initHidden(self, ctx):
#         return [F.zeros((1, 1, self.hidden_size), ctx=ctx)]
#
# class Decoder(Block):
#     def __init__(self, max_labels, context_size):
#         super(Decoder, self).__init__()
#         self.max_labels = max_labels
#         with self.name_scope():
#             self.out = nn.Dense(max_labels, in_units=context_size)
#
#     def forward(self, hidden):
#         output = self.out(hidden[0])
#         return output


class ClassifierRNN(Block):
    def __init__(self, word_embedding, max_labels, hidden_size):
        super(ClassifierRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = word_embedding.idx_to_vec.shape[1]
        with self.name_scope():
            # self.embedding = word_embedding #nn.Embedding(input_size, hidden_size)
            self.embedding = emb.layer_for_embedding(word_embedding)

            # FREEZE the embedding layer
            # self.embedding.collect_params().setattr('grad_req', 'null')
            # TODO: assert somehow that the output size is the same
            # could use 'infer_shape' https://github.com/apache/incubator-mxnet/issues/1090

            self.rnn = rnn.GRU(hidden_size,
                               input_size=self.embedding_size)
            self.out = nn.Dense(max_labels, in_units=hidden_size
                                #self.embedding_size
                                )

    def forward(self, input, hidden):
        output = self.embedding(input).swapaxes(0, 1)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden(self, ctx):
        return [F.zeros((1, 1, self.hidden_size), ctx=ctx)]


def train(input_variable,  # single sequence
          target_variable,
          classifier,
          # decoder,
          classifier_optimizer,
          # decoder_optimizer,
          criterion, max_length, ctx):

    with autograd.record():
        loss = F.zeros((1,), ctx=ctx)

        input_length = input_variable.shape[0]
        target_length = target_variable.shape[0]

        classifier_hidden = classifier.init_hidden(ctx)
        classifier_outputs, classifier_hidden = classifier(
            input_variable.expand_dims(0), classifier_hidden)

        #         decoder_input = F.array([SOS_token], ctx=ctx) # NOTE: issue here
        #         decoder_hidden = encoder_hidden

        #         decoder_outputs, decoder_hidden = decoder(
        #                 target_variable.expand_dims(0), decoder_hidden)
        # decoder_outputs = decoder(encoder_hidden)

        # print('enc_out.shape:', classifier_outputs.shape)
        # print('enc_hidden:', classifier_hidden)


        for di in range(target_length):
            # loss = F.add(loss,
            #              criterion(decoder_outputs[di], target_variable[di]))
            loss = F.add(loss,
                         criterion(classifier_outputs[di],
                                   target_variable[di]))
            # print(criterion(decoder_outputs[di], target_variable[di]))

        loss.backward()

    classifier_optimizer.step(1)
    # decoder_optimizer.step(1)

    return loss.asscalar() / target_length


def train_iters(training_pairs: List[nd.array],
                classifier,
                # decoder,
                ctx,
                print_every=100,
                print_loss_total=0,  # Reset every print_every
                epochs=100,

                ):

    classifier.initialize(ctx=ctx)
    # decoder.initialize(ctx=ctx)

    classifier_optimizer = gluon.Trainer(classifier.collect_params(), 'sgd', {'learning_rate': 0.1})
    # decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'sgd', {'learning_rate': 0.1})

#     training_pairs = [variablesFromPair(random.choice(pairs))
#                       for i in range(num_iterations)]

    criterion = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)

    for epoch in range(epochs):
        for iter in range(1, len(training_pairs)):
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0].as_in_context(ctx)
            target_variable = training_pair[1].as_in_context(ctx)

            loss = train(input_variable, target_variable,
                         classifier=classifier,
                         #decoder,
                         classifier_optimizer=classifier_optimizer,
                         #decoder_optimizer,
                         criterion=criterion,
                         max_length=32,
                         ctx=ctx)

            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print("epoch", epoch, "iteration", iter, " avg loss: ", \
                                print_loss_avg)


def eval(test_pairs: List[nd.array],
         classifier, ctx,
         debug=False):

    if debug:
        log = print
    else:
        log = lambda *_args, **_kwargs: None

    total_words = 0
    # word_errors = 0
    total_examples = len(test_pairs)
    example_errors = set()
    word_errors = 0    # TODO make this a dict so we can derive a confusion
    # matrix easily

    for tp in test_pairs:
        input_variable = tp[0].as_in_context(ctx)
        target_variable = tp[1].as_in_context(ctx)

        target_length = target_variable.shape[0]
        classifier_hidden = classifier.init_hidden(ctx)

        classifier_outputs, classifier_hidden = classifier(
            input_variable.expand_dims(0), classifier_hidden)

        log("***************************************************************")
        log("testing pair", tp)
        for di in range(target_length):

            total_words += 1
            predicted_output = nd.argmax(classifier_outputs[di], axis=0)
            correct_output = target_variable[di]

            log(int(predicted_output.asscalar()), correct_output.asscalar())

            if int(predicted_output.asscalar()) != correct_output.asscalar():
                log("prediction failed. raw classifier output:")
                log(classifier_outputs[di])
                word_errors += 1
                example_errors.add(tp)



    return {'total_examples':total_examples,
            'total_example_errors': len(example_errors),
            'example_errors': example_errors,
            'total_words': total_words,
            'word_errors': word_errors}
