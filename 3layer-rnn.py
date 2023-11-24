import numpy as np

# data I/O
data = open('input.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has {} characters, {} unique.'.format(data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 512  # size of hidden layer of neurons
hidden_size_2 = 512  # size of second hidden layer of neurons
hidden_size_3 = 512  # size of third hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-2

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Wxh2 = np.random.randn(hidden_size_2, hidden_size) * 0.01  # hidden to second hidden
Whh2 = np.random.randn(hidden_size_2, hidden_size_2) * 0.01  # second hidden to second hidden
Wxh3 = np.random.randn(hidden_size_3, hidden_size_2) * 0.01  # second hidden to third hidden
Whh3 = np.random.randn(hidden_size_3, hidden_size_3) * 0.01  # third hidden to third hidden
Why = np.random.randn(vocab_size, hidden_size_3) * 0.01  # third hidden to output

bh = np.zeros((hidden_size, 1))  # hidden bias
bh2 = np.zeros((hidden_size_2, 1))  # second hidden bias
bh3 = np.zeros((hidden_size_3, 1))  # third hidden bias
by = np.zeros((vocab_size, 1))  # output bias

def lossFun(inputs, targets, hprev, hprev2, hprev3):
    """
    inputs,targets are both list of integers.
    hprev, hprev2, hprev3 are Hx1 array of initial hidden state for each layer
    returns the loss, gradients on model parameters, and last hidden state for each layer
    """
    xs, hs, hs2, hs3, ys, ps = {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    hs2[-1] = np.copy(hprev2)
    hs3[-1] = np.copy(hprev3)
    loss = 0

    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
        hs2[t] = np.tanh(np.dot(Wxh2, hs[t]) + np.dot(Whh2, hs2[t - 1]) + bh2)  # second hidden state
        hs3[t] = np.tanh(np.dot(Wxh3, hs2[t]) + np.dot(Whh3, hs3[t - 1]) + bh3)  # third hidden state
        ys[t] = np.dot(Why, hs3[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWxh2, dWhh2, dWxh3, dWhh3, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Wxh2), np.zeros_like(Whh2), np.zeros_like(Wxh3), np.zeros_like(Whh3), np.zeros_like(Why)
    dbh, dbh2, dbh3, dby = np.zeros_like(bh), np.zeros_like(bh2), np.zeros_like(bh3), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    dhnext2 = np.zeros_like(hs2[0])
    dhnext3 = np.zeros_like(hs3[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs3[t].T)
        dby += dy

        dh3 = np.dot(Why.T, dy) + dhnext3  # backprop into third hidden layer
        dh3raw = (1 - hs3[t] * hs3[t]) * dh3  # backprop through tanh nonlinearity
        dbh3 += dh3raw
        dWxh3 += np.dot(dh3raw, hs2[t].T)
        dWhh3 += np.dot(dh3raw, hs3[t - 1].T)
        dhnext3 = np.dot(Whh3.T, dh3raw)

        dh2 = np.dot(Wxh3.T, dh3raw) + dhnext2  # backprop into second hidden layer
        dh2raw = (1 - hs2[t] * hs2[t]) * dh2  # backprop through tanh nonlinearity
        dbh2 += dh2raw
        dWxh2 += np.dot(dh2raw, hs[t].T)
        dWhh2 += np.dot(dh2raw, hs2[t - 1].T)
        dhnext2 = np.dot(Whh2.T, dh2raw)

        dh = np.dot(Wxh2.T, dh2raw) + dhnext  # backprop into first hidden layer
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWxh2, dWhh2, dWxh3, dWhh3, dWhy, dbh, dbh2, dbh3, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWxh2, dWhh2, dWxh3, dWhh3, dWhy, dbh, dbh2, dbh3, dby, hs[len(inputs) - 1], hs2[len(inputs) - 1], hs3[len(inputs) - 1]

# Sample function remains the same but uses the last hidden state of the third layer
def sample(h, h2, h3, seed_ix, n):
    """
    sample a sequence of integers from the model
    h, h2, h3 are memory states for each layer, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        h2 = np.tanh(np.dot(Wxh2, h) + np.dot(Whh2, h2) + bh2)
        h3 = np.tanh(np.dot(Wxh3, h2) + np.dot(Whh3, h3) + bh3)
        y = np.dot(Why, h3) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

# Everything below is the same, but we need to initialize the memory for the additional layers and include them in the parameter updates
n, p = 0, 0
mWxh, mWhh, mWxh2, mWhh2, mWxh3, mWhh3, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Wxh2), np.zeros_like(Whh2), np.zeros_like(Wxh3), np.zeros_like(Whh3), np.zeros_like(Why)
mbh, mbh2, mbh3, mby = np.zeros_like(bh), np.zeros_like(bh2), np.zeros_like(bh3), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        hprev2 = np.zeros((hidden_size_2, 1))  # reset second hidden layer RNN memory
        hprev3 = np.zeros((hidden_size_3, 1))  # reset third hidden layer RNN memory
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, hprev2, hprev3, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n {} \n----'.format(txt))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWxh2, dWhh2, dWxh3, dWhh3, dWhy, dbh, dbh2, dbh3, dby, hprev, hprev2, hprev3 = lossFun(inputs, targets, hprev, hprev2, hprev3)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print('iter {}, loss: {}'.format(n, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Wxh2, Whh2, Wxh3, Whh3, Why, bh, bh2, bh3, by],
                                  [dWxh, dWhh, dWxh2, dWhh2, dWxh3, dWhh3, dWhy, dbh, dbh2, dbh3, dby],
                                  [mWxh, mWhh, mWxh2, mWhh2, mWxh3, mWhh3, mWhy, mbh, mbh2, mbh3, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter 
