#! /usr/bin/python3

import argparse
import torch

# Mostly the example network from pytorch docs
class Net(torch.nn.Module):
    """Network that demonstrates some fundamentals of fitting."""

    def __init__(self, num_inputs, num_layers, dropout_layer=False, expand_layers=False):
        """Initialize the demonstration network with a single output.

        Arguments:
            num_inputs    (int): Number of inputs.
            num_layers    (int): Number of layers before the output layer. Must be > 0. Each layer will
                                 have num_inputs units.
            dropout_layer(bool): Use dropout after the given ReLU. No dropout if 0.
            expand_layers(bool): Double layer size with each linear layer.
        """
        super(Net, self).__init__()
        self.net = torch.nn.ModuleList()
        layer_widths = [num_inputs] * (num_layers)
        # Expand the hidden layers if requested
        if expand_layers:
            for i in range(1, len(layer_widths)):
                layer_widths[i] *= 2
        for i in range(num_layers - 1):
            self.net.append(torch.nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if num_layers > 1:
                self.net.append(torch.nn.ELU())
            else:
                self.net.append(torch.nn.ReLU())
            if (dropout_layer - 1) == i:
                self.net.append(torch.nn.Dropout(p=0.25))
        self.net.append(torch.nn.Linear(layer_widths[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

def getBatch(batch_size, correlations, anticorrelations):
    """Make inputs and outputs for a single output model.

    Arguments:

        batch_size (int)               : Number of elements in a batch.
        correlations     (torch.tensor): Probabilities of each input being 1 when the target output
                                         is 1.
        anticorrelations (torch.tensor): Probabilities of each input being 1 when the target output
                                         is 0. Must be the same length as correlations.
    """
    assert len(correlations) == len(anticorrelations)
    # Make some 0s and 1s as the output
    outputs = torch.empty((batch_size, 1)).random_(0, 2).cuda()

    # Randomly select which inputs would be active if the output is 0 or 1
    probs = correlations.expand(batch_size, -1) * outputs + anticorrelations.expand(batch_size, -1) * (1. - outputs)
    return torch.bernoulli(probs), outputs


inparser = argparse.ArgumentParser(
    description="Arguments for the noise training script.")
inparser.add_argument(
    '--layers', type=int, default=1,
    help='Number of intermediate layers in the model.')
inparser.add_argument(
    '--correlations', type=float, nargs='+', required=True,
    help='Input probability of being 1 when the output is 1.')
inparser.add_argument(
    '--anticorrelations', type=float, nargs='+', required=True,
    help='Input probability of being 1 when the output is 0.')
inparser.add_argument(
    '--dropout', default=0, type=int,
    help='Enable dropout after the given ReLU layer. 0 to disable.')
inparser.add_argument(
    '--expand', default=False, action='store_true',
    help='Expand the size of linear layers, doubling the parameters at each subsequent layer.')
args = inparser.parse_args()

net = Net(num_inputs=len(args.correlations),
          num_layers=args.layers,
          dropout_layer=args.dropout,
          expand_layers=args.expand).cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_fn = torch.nn.L1Loss()

correlations = torch.tensor(args.correlations).cuda()
anticorrelations = torch.tensor(args.anticorrelations).cuda()

max_batches = 5000
for batch_num in range(max_batches):
    batch, labels = getBatch(batch_size=32, correlations=correlations, anticorrelations=anticorrelations)
    optimizer.zero_grad()
    out = net.forward(batch)
    loss = loss_fn(out, labels)
    if 0 == batch_num % 100:
        with torch.no_grad():
            print(f"Batch {batch_num} loss is {loss.mean()}")
            if 1 == args.layers:
                for i, layer in enumerate(net.net):
                    if hasattr(layer, 'weight'):
                        print(f"At batch {batch_num} layer {i} has weights {layer.weight.tolist()[0]} and bias {layer.bias.tolist()[0]}")
            else:
                # Just check the correlation
                net.eval()
                for i in range(len(correlations)):
                    test_in = torch.zeros(1, len(correlations)).cuda()
                    test_in[0,i] = 1.
                    test_out = net.forward(test_in)
                    print(f"At batch {batch_num} input {i} has correlation {test_out[0].item()}")
                net.train()
    loss.backward()
    optimizer.step()

batch_num = max_batches
with torch.no_grad():
    print(f"Batch {batch_num} loss is {loss.mean()}")
    if 1 == args.layers:
        for i, layer in enumerate(net.net):
            if hasattr(layer, 'weight'):
                print(f"At batch {batch_num} layer {i} has weights {layer.weight.tolist()[0]} and bias {layer.bias.tolist()[0]}")
    else:
        # Just check the correlation
        net.eval()
        for i in range(len(correlations)):
            test_in = torch.zeros(1, len(correlations)).cuda()
            test_in[0,i] = 1.
            test_out = net.forward(test_in)
            print(f"At batch {batch_num} input {i} has correlation {test_out[0].item()}")
        net.train()
