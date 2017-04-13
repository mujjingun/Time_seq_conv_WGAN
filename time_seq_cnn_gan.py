# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.distributions as tfcd
import numpy as np
import random, itertools, os

load_from = None#'./models/ch/'
save_to = './models/cht/'
log_dir = './log/tflogs3'

seq_size = 1000
voca = " 0123456789abcdefghijklmnopqrstuvwxyz" \
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ.!;:,?&$'-\n"
dic = {voca[i]: i for i in range(len(voca))}
num_layers = 10
filt_size = 5
hidden_size = 200
batch_size = 5
num_critic = 1
clip = 0.01

def normalize(x, axis, prefix="", eps=1e-05, beta=0.1):
    var_shape = [x.get_shape()[axis]]
    gain = tf.get_variable(prefix + "gain", var_shape,
    initializer=tf.constant_initializer(1))
    bias = tf.get_variable(prefix + "bias", var_shape,
    initializer=tf.constant_initializer(0))
    mean, var = tf.nn.moments(x, [axis], keep_dims=True)
    out = (x - mean) / tf.sqrt(var + eps) * beta
    out = out * gain + bias
    return out

def init_with_variance(var):
    return tf.random_normal_initializer(stddev=np.sqrt(var))

# Model
def build_gen_model(x, z, state=None, reuse=False):
    with tf.variable_scope("gen", reuse=reuse):
        with tf.variable_scope("pre"):
            x = tf.concat([x, z], 2)
            filt = tf.get_variable("filt", [1, len(voca) + 1, hidden_size * 2],
                initializer=tf.random_normal_initializer())
            skip = tf.nn.convolution(x, filt, "VALID")
        for n in range(num_layers):
            with tf.variable_scope("main{}".format(n)):
                r = 2 ** n
                l = r * (filt_size - 1) + 1
                norm = normalize(skip, 2)
                # Gated Linear Unit
                A = norm[:, :, :hidden_size]
                B = norm[:, :, hidden_size:]
                act = A * tf.sigmoid(B)
                # Atrous Convolution
                filt = tf.get_variable("filt",
                    [filt_size, hidden_size, hidden_size * 2],
                    initializer=init_with_variance(2 / hidden_size))
                act = tf.pad(act, [[0, 0], [l - 1, 0], [0, 0]])
                conv = tf.nn.convolution(act, filt, "VALID", dilation_rate=[r])
                # Residual Skip Connection
                skip += conv
        with tf.variable_scope("post{}".format(n)):
            norm = normalize(skip, 2)
            # ReLU Activation
            act = tf.concat([tf.nn.relu(norm), tf.nn.relu(-norm)], 2)
            # 1x1 Convolution
            filt = tf.get_variable("filt",
                [1, hidden_size * 4, len(voca)],
                initializer=init_with_variance(1 / hidden_size))
            conv = tf.nn.convolution(act, filt, "VALID")
            out = normalize(conv, 2, "final")
    if state is not None:
        return out, state
    else:
        return out

def build_critic_model(x, y, reuse=False):
    with tf.variable_scope("critic", reuse=reuse):
        with tf.variable_scope("pre"):
            x = tf.concat([x, y], 2)
            filt = tf.get_variable("filt", [1, len(voca) * 2, hidden_size * 2],
                initializer=tf.random_normal_initializer())
            skip = tf.nn.convolution(x, filt, "VALID")
        for n in range(num_layers):
            with tf.variable_scope("main{}".format(n)):
                r = 2 ** n
                l = r * (filt_size - 1) + 1
                norm = normalize(skip, 2)
                # Gated Linear Unit
                A = norm[:, :, :hidden_size]
                B = norm[:, :, hidden_size:]
                act = A * tf.sigmoid(B)
                # Atrous Convolution
                filt = tf.get_variable("filt",
                    [filt_size, hidden_size, hidden_size * 2],
                    initializer=init_with_variance(2 / hidden_size))
                act = tf.pad(act, [[0, 0], [l - 1, 0], [0, 0]])
                conv = tf.nn.convolution(act, filt, "VALID", dilation_rate=[r])
                # Residual Skip Connection
                skip += conv
        with tf.variable_scope("post{}".format(n)):
            norm = normalize(skip, 2)
            # ReLU Activation
            act = tf.nn.relu(norm)
            # 1x1 Convolution
            filt = tf.get_variable("filt",
                [1, hidden_size * 2, hidden_size],
                initializer=init_with_variance(1 / hidden_size))
            conv = tf.nn.convolution(act, filt, "VALID")
            # Average up
            avg = normalize(tf.reduce_mean(conv, axis=1), 1, "avg")
            act = tf.nn.relu(avg)
            W = tf.get_variable("W", [hidden_size, 1],
                initializer=init_with_variance(1 / hidden_size))
            out = tf.matmul(act, W)
    return out

global_step = tf.get_variable("step", [], initializer=tf.zeros_initializer())
data = tf.placeholder(tf.int32, [None, seq_size], "data")
noise = tf.random_normal([tf.shape(data)[0], seq_size - 1, 1])
one_hot = tf.one_hot(data, len(voca))
x = one_hot[:, :-1]
y = one_hot[:, 1:]

# Critic
real = build_critic_model(x, y)
c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
clipped = [p.assign(tf.clip_by_value(p, -clip, clip)) for p in c_params]

with tf.control_dependencies(clipped):
    gen = build_gen_model(x, noise)
    C = tf.reduce_mean(build_critic_model(x, gen, True) - real)
    c_optimizer =tf.train.RMSPropOptimizer(1e-4)
    c_train_step = c_optimizer.minimize(C, var_list=c_params)

c_summ = tf.summary.merge([
    tf.summary.scalar('Critic', C),
    ])

# Generator
gen = build_gen_model(x, noise, None, True)
G = tf.reduce_mean(-build_critic_model(x, gen, True))

g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gen")
g_optimizer = tf.train.RMSPropOptimizer(1e-4)
g_train_step = g_optimizer.minimize(G, global_step, g_params)

sample = tf.argmax(gen[0], 1)
g_summ = tf.summary.merge([
    tf.summary.scalar('Generator', G),
    ])

# Inference Model
state = tf.placeholder(tf.float32, [num_layers, None, hidden_size], "state")

seed = tf.constant([0.0] * (len(voca) - 1) + [1.0]) # Start with a newline
seed = seed[None, None, :]
noise = tf.random_normal([1, 1, 1])
out, state = build_gen_model(seed, noise, state, True)

# Training
saver = tf.train.Saver()
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.graph.finalize()

train_writer = tf.summary.FileWriter(log_dir, sess.graph)

if load_from is not None:
    print("Loading from {}".format(load_from))
    saver.restore(sess, tf.train.latest_checkpoint(load_from))

with open('shakespeare.txt') as f:
    dataset = f.read()
    dataset = [dic[c] for c in dataset]

while True:
    # Train Generator
    s = random.sample(range(len(dataset) - seq_size), batch_size)
    feed = {data: [dataset[s:s + seq_size] for s in s]}
    evals = [g_train_step, G, g_summ, global_step, sample]
    _, g, summ, step, s = sess.run(evals, feed_dict=feed)
    train_writer.add_summary(summ, step)
    print(step, "g", g)
    print("".join(voca[n] for n in s))

    # Train Critic
    for i in range(num_critic):
        s = random.sample(range(len(dataset) - seq_size), batch_size)
        feed = {data: [dataset[s:s + seq_size] for s in s]}
        _, c, summ = sess.run([c_train_step, C, c_summ], feed_dict=feed)
    train_writer.add_summary(summ, step)
    print(step, c)

    # Save model every 500 iterations
    if step % 500 == 0:
        saver.save(sess, os.path.join(save_to, 'model'))
        print("Saved.")

### END OF FILE
