# Created by Geon Park (mujjingun@gmail.com)
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.distributions as tfcd
import numpy as np
import random, itertools, os

load_from = None
save_to = '../models/chargan_block/'
#load_from = save_to
log_dir = '../log/tflogs6/'

voca = " 0123456789abcdefghijklmnopqrstuvwxyz" \
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ.!;:,?&$'-\n"
dic = {voca[i]: i for i in range(len(voca))}
num_layers = 9
filt_size = 7
input_size = (2 ** (num_layers - 1) - 1) * (filt_size - 1) + 1
output_size = (2 ** (num_layers - 1)) * (filt_size - 1)
seq_size = input_size + output_size
hidden_size = 200
num_post = 3
batch_size = 2
num_critic = 1
clip = 0.01
learning_rate = 5e-5

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
def build_gen_model(x):
    with tf.variable_scope("pre"):
        filt = tf.get_variable("filt", [1, len(voca), hidden_size * 2],
            initializer=tf.random_normal_initializer())
        skip = tf.nn.convolution(x, filt, "VALID")
    for n in range(num_layers - 1):
        with tf.variable_scope("input{}".format(n)):
            r = 2 ** n
            l = r * (filt_size - 1)
            norm = normalize(skip, 2)
            # Gated Linear Unit
            A = norm[:, :, :hidden_size]
            B = norm[:, :, hidden_size:]
            act = A * tf.sigmoid(B)
            # Atrous Convolution
            act = tf.pad(act, [[0, 0], [l, 0], [0, 0]])
            filt = tf.get_variable("filt",
                [filt_size, hidden_size, hidden_size * 2],
                initializer=init_with_variance(2 / hidden_size))
            conv = tf.nn.convolution(act, filt, "VALID", dilation_rate=[r])
            # Residual Skip Connection
            skip += conv
    skip = tf.reduce_mean(skip, 1, keep_dims=True)
    skip = tf.tile(skip, [1, output_size, 1])
    for m in range(3):
        for n in range(num_layers):
            with tf.variable_scope("main{}.{}".format(m, n)):
                r = 2 ** n
                l = r * (filt_size - 1)
                noise = tf.random_normal(tf.shape(skip))
                norm = normalize(skip, 2) * noise
                # Gated Linar Unit
                A = norm[:, :, :hidden_size]
                B = norm[:, :, hidden_size:]
                act = A * tf.sigmoid(B)
                # Atrous Convolution
                filt = tf.get_variable("filt",
                    [filt_size, hidden_size, hidden_size * 2],
                    initializer=init_with_variance(2 / hidden_size))
                conv = tf.nn.convolution(act, filt, "SAME", dilation_rate=[r])
                # Residual Skip Connection
                skip += conv
    with tf.variable_scope("post"):
        norm = normalize(skip, 2)
        # ReLU activation
        act = tf.concat([tf.nn.relu(norm), tf.nn.relu(-norm)], 2)
        # 1x1 Convolution
        filt = tf.get_variable("filt",
            [1, hidden_size * 4, hidden_size],
            initializer=init_with_variance(1 / hidden_size))
        conv = tf.nn.convolution(act, filt, "VALID")
        norm = normalize(conv, 2, "norm2")
        act = tf.concat([tf.nn.relu(norm), tf.nn.relu(-norm)], 2)
        # Second 1x1 Convolution
        filt2 = tf.get_variable("filt2",
            [1, hidden_size * 2, len(voca)],
            initializer=init_with_variance(1 / hidden_size))
        conv = tf.nn.convolution(act, filt2, "VALID")
        # Normalize Variance
        mean, var = tf.nn.moments(conv, [2], keep_dims=True)
        eps = 1e-05
        std = tf.sqrt(var + eps)
        exp_mean = 1 / len(voca)
        exp_dev = np.sqrt(len(voca) - 1) / len(voca)
        out = (conv - mean + exp_mean) / std * exp_dev
    return out

def build_critic_model(x, y):
    with tf.variable_scope("pre"):
        x = tf.concat([x, y], 1)
        filt = tf.get_variable("filt", [1, len(voca), hidden_size * 2],
            initializer=tf.random_normal_initializer())
        skip = tf.nn.convolution(x, filt, "VALID")
    for n in range(num_layers):
        with tf.variable_scope("main{}".format(n)):
            r = 2 ** n
            l = r * (filt_size - 1)
            norm = normalize(skip, 2)
            # Gated Linear Unit
            A = norm[:, :, :hidden_size]
            B = norm[:, :, hidden_size:]
            act = A * tf.sigmoid(B)
            # Atrous Convolution
            act = tf.pad(act, [[0, 0], [l, 0], [0, 0]])
            filt = tf.get_variable("filt",
                [filt_size, hidden_size, hidden_size * 2],
                initializer=init_with_variance(2 / hidden_size))
            conv = tf.nn.convolution(act, filt, "VALID", dilation_rate=[r])
            # Residual Skip Connection
            skip += conv
    with tf.variable_scope("out"):
        skip = normalize(skip, 2)
        mean = tf.reduce_mean(skip, 1)
        act = tf.concat([tf.nn.relu(mean), tf.nn.relu(-mean)], 1)
        W1 = tf.get_variable("W1", [hidden_size * 4, hidden_size],
            initializer=init_with_variance(2 / (hidden_size * 4)))
        y1 = normalize(tf.matmul(act, W1), 1, "norm2")
        act = tf.concat([tf.nn.relu(y1), tf.nn.relu(-y1)], 1)
        W2 = tf.get_variable("W2", [hidden_size * 2, 1],
            initializer=init_with_variance(2 / (hidden_size * 2)))
        y2 = tf.matmul(act, W2)
    return y2

global_step = tf.get_variable("step", [], initializer=tf.zeros_initializer())
data = tf.placeholder(tf.int32, [None, seq_size], "data")
one_hot = tf.one_hot(data, len(voca))
x = one_hot[:, :input_size]
y = one_hot[:, input_size:]

build_gen_model = tf.make_template('gen', build_gen_model)
build_critic_model = tf.make_template('critic', build_critic_model)

# Critic
real = tf.reduce_mean(build_critic_model(x, y))
c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
with tf.variable_scope('clip'):
    clipped = [p.assign(tf.clip_by_value(p, -clip, clip)) for p in c_params]

with tf.control_dependencies(clipped):
    gen = build_gen_model(x)
    fake = tf.reduce_mean(build_critic_model(x, gen))
    C = fake - real
    c_optimizer = tf.train.RMSPropOptimizer(learning_rate)
    c_train_step = c_optimizer.minimize(C, var_list=c_params)

c_summ = tf.summary.merge([
    tf.summary.scalar('Critic', C),
    tf.summary.scalar('Real', real),
    tf.summary.scalar('Fake', fake)
    ])

# Generator
gen = build_gen_model(x)
G = tf.reduce_mean(-build_critic_model(x, gen))

g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gen")
g_optimizer = tf.train.RMSPropOptimizer(learning_rate)
g_train_step = g_optimizer.minimize(G, global_step, g_params)

sample = tf.argmax(gen[0], 1)
g_summ = tf.summary.merge([
    tf.summary.scalar('Generator', G),
    ])

# Inference Model
#seed = tf.placeholder(tf.int32, [1], "seed")
#seed_one_hot = tf.one_hot(seed[None, :], len(voca))
#noise = tf.random_normal([1, 1, 1])
#pred, next_state = build_gen_model(seed_one_hot, noise)

# Training
saver = tf.train.Saver()
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

cpu_config = tf.ConfigProto(device_count = {'GPU': 0})
cpu_sess = tf.Session(config=cpu_config)
cpu_sess.run(tf.global_variables_initializer())

sess.graph.finalize()
cpu_sess.graph.finalize()

train_writer = tf.summary.FileWriter(log_dir, sess.graph)

if load_from is not None:
    print("Loading from {}".format(load_from))
    chkp = tf.train.latest_checkpoint(load_from)
    saver.restore(sess, chkp)
    saver.restore(cpu_sess, chkp)
    print("Continuing from iteration", cpu_sess.run([global_step]))

def sample_from_model(length):
    d = np.zeros([1])
    s = np.zeros([num_layers, state_size, hidden_size])
    for i in range(length):
        feed = {seed: d[-1:], state: [s]}
        dn, sn = cpu_sess.run([pred, next_state], feed_dict=feed)
        # dn: (batch, seq, voca_size)
        # sn: (num_layers, batch, state_size, hidden_size)
        dn = np.argmax(dn[0], 1)
        print(i, dn)
        d = np.append(d, dn, axis=0)
        s = np.append(s[:, 1:], sn[:, 0], axis=1)
    return "".join(voca[int(n)] for n in d)
#print(sample_from_model(100))

with open('shakespeare.txt') as f:
    dataset = f.read()
    dataset = [dic[c] for c in dataset]

def fetch_data():
    s = random.sample(range(len(dataset) - seq_size), batch_size)
    feed = {data: [dataset[s:s + seq_size] for s in s]}
    return feed

while True:
    # Train Generator
    feed = fetch_data()
    evals = [g_train_step, G, g_summ, global_step, sample]
    _, g, summ, step, s = sess.run(evals, feed_dict=feed)
    train_writer.add_summary(summ, step)
    print(step, "g", g)
    print("".join(voca[n] for n in s))

    # Train Critic
    for i in range(num_critic):
        feed = fetch_data()
        _, c, summ = sess.run([c_train_step, C, c_summ], feed_dict=feed)
    train_writer.add_summary(summ, step)
    print(step, c)

    # Save model every 500 iterations
    if save_to is not None and step % 500 == 0:
        saver.save(sess, os.path.join(save_to, 'model'))
        print("Saved.")

### END OF FILE
