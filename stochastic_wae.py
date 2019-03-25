import functools
import tensorflow as tf
import tfsnippet as spt
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from argparse import ArgumentParser
import sys
import numpy as np
from tfsnippet.examples.utils import bernoulli_flow
import os
from tfsnippet.examples.utils.evaluation import save_images_collection
from utils import Logger
import time

class config(spt.Config):
    shape=(28,28,1)
    d_z = 8
    gan_learning_rate = 0.0005
    reconstruction_learning_rate = 0.001
    lr_annealing_factor = 1.
    reg_lambda = 10.
    batch_size = 100
    test_batch_size = 500
    test_epoch_freq = 10
    plot_epoch_freq = 10
    adam_beta1 = 0.5
    adam_beta2 = 0.999
    plotting_shape = (10,10)
    plotting_number = 100
    max_epoch = 100
    conv_filters = 4#128
    grad_clip_norm=None

train_config=config()


@spt.global_reuse
@add_arg_scope
def q_net(x, observed=None, is_initializing=False):
    net = spt.BayesianNet(observed=observed)
    with arg_scope([spt.layers.conv2d],
                   padding='same',
                   strides=(2, 2),
                   activation_fn=tf.nn.relu,
                   normalizer_fn=functools.partial(
                            spt.layers.act_norm,
                            axis=-1,
                            value_ndims=3,
                            initializing=is_initializing,
                        )
                   ):
        x = tf.to_float(x)
        q_z = spt.layers.conv2d(input=x, out_channels=train_config.conv_filters, kernel_size=4, name="q_net_1")
        q_z = spt.layers.conv2d(input=q_z, out_channels=train_config.conv_filters*2, kernel_size=4, name="q_net_2")
        q_z = spt.layers.conv2d(input=q_z, out_channels=train_config.conv_filters*4, kernel_size=4, name="q_net_3")
        q_z = spt.layers.conv2d(input=q_z, out_channels=train_config.conv_filters*8, kernel_size=4, name="q_net_4")
        q_z = spt.ops.reshape_tail(q_z,3,[-1], name="q_net_reshape")
    z_mean = spt.layers.dense(input=q_z, units=train_config.d_z, name="q_net_z_mean", kernel_initializer=tf.zeros_initializer)
    z_logstd = spt.layers.dense(input=q_z, units=train_config.d_z, name="q_net_z_logstd", kernel_initializer=tf.zeros_initializer)
    z = net.add("q_net_z",spt.Normal(mean=z_mean, logstd=z_logstd),group_ndims=1)
    return net

@spt.global_reuse
@add_arg_scope
def p_net(z,observed=None, is_initializing=False):
    net = spt.BayesianNet(observed=observed)
    with arg_scope([spt.layers.conv2d, spt.layers.deconv2d],
                   strides=(2,2),
                   padding='same',
                   activation_fn=tf.nn.relu,
                   normalizer_fn=functools.partial(
                                  spt.layers.act_norm,
                                  axis=-1,
                                  value_ndims=3,
                                  initializing=is_initializing,
                              )
                   ):
        p_x = spt.layers.dense(input=z, units=7*7*train_config.conv_filters*8, name="p_net_1")
        p_x = spt.ops.reshape_tail(input=p_x, ndims=1, shape=(7,7,train_config.conv_filters*8), name="p_net_2")
        p_x = spt.layers.deconv2d(input=p_x, out_channels=train_config.conv_filters*4, kernel_size=4, name="p_net_3")
        p_x = spt.layers.deconv2d(input=p_x, out_channels=train_config.conv_filters*2, kernel_size=4, name="p_net_4")
    x_logits = spt.layers.conv2d(input=p_x, out_channels=1, kernel_size=4, strides=(1,1),
                            padding='same', name="p_net_x_logits", kernel_initializer=tf.zeros_initializer)
    p_x = net.add("p_net_x",spt.Bernoulli(logits=x_logits, dtype=tf.float32),group_ndims=3)
    return net

@spt.global_reuse
@add_arg_scope
def d_net(z,is_initializing=False):
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.relu,
                   normalizer_fn=functools.partial(
                                  spt.layers.act_norm,
                                  axis=-1,
                                  value_ndims=3,
                                  initializing=is_initializing,
                              )
                   ):
        d = spt.layers.dense(input=z, units=512, name="d_net_1")
        d = spt.layers.dense(input=d, units=512, name="d_net_2")
        d = spt.layers.dense(input=d, units=512, name="d_net_3")
        d = spt.layers.dense(input=d, units=512, name="d_net_4")
    d = spt.layers.dense(input=d, units=1, name="d_net_output")
    return d

def rand_z(size):
    return np.random.normal(loc=np.zeros(train_config.d_z), scale=np.ones(train_config.d_z),size=(size,) + (train_config.d_z,))

def main():
    res_dir='results/wae_mnist'
    plotting_dir = os.path.join(res_dir, 'plotting')
    if not os.path.isdir(plotting_dir):
        os.makedirs(plotting_dir)
    sys.stdout = Logger(os.path.join(res_dir,'console.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(res_dir,'error.log'), sys.stderr)
    arg_parser = ArgumentParser()
    spt.register_config_arguments(train_config, arg_parser, title='train config')
    arg_parser.parse_args(sys.argv[1:])
    x=tf.placeholder(tf.float32,shape=(None,) + train_config.shape,name="x")
    gan_learning_rate=tf.placeholder(tf.float32, shape=None, name='gan_learning_rate')
    reconstruction_learning_rate=tf.placeholder(tf.float32, shape=None, name='reconstruction_learning_rate')
    random_z = tf.placeholder(tf.float32,shape=(None,)+(train_config.d_z,), name='random_z')

    def _gan_loss(dqz, dpz):
        loss = -tf.nn.softplus(-dpz)-dqz-tf.nn.softplus(-dqz)
        return tf.reduce_mean(train_config.reg_lambda * loss)

    def _reconstruction_loss(x, logits, dqz):
        recons = tf.sigmoid(logits)
        distance = tf.reduce_sum(tf.square(x - recons) + recons*(1.-recons), axis=[1,2])
        loss = -tf.nn.softplus(-dqz)
        return tf.reduce_mean(distance - train_config.reg_lambda * loss)

    with tf.name_scope('initializing'), \
            arg_scope([p_net, q_net, d_net], is_initializing=True):
        init_qz = q_net(x)["q_net_z"]
        init_dqz = d_net(init_qz)
        init_dpz = d_net(random_z)
        init_logits = p_net(init_qz)["p_net_x"].distribution.logits
        init_gan_loss = _gan_loss(init_dqz, init_dpz)
        init_reconstruction_loss = _reconstruction_loss(x, init_logits, init_dqz)

    with tf.name_scope('training'), \
            arg_scope([p_net, q_net, d_net], is_initializing=False):
        train_qz = q_net(x)["q_net_z"]
        train_dqz = d_net(train_qz)
        train_dpz = d_net(random_z)
        train_logits = p_net(train_qz)["p_net_x"].distribution.logits
        train_gan_loss = _gan_loss(train_dqz, train_dpz)
        train_reconstruction_loss = _reconstruction_loss(x, train_logits, train_dqz)

    with tf.name_scope('testing'), \
            arg_scope([p_net, q_net, d_net], is_initializing=False):
        test_qz = q_net(x)["q_net_z"]
        test_dqz = d_net(test_qz)
        test_dpz = d_net(random_z)
        test_logits = p_net(test_qz)["p_net_x"].distribution.logits
        test_gan_loss = _gan_loss(test_dqz, test_dpz)
        test_reconstruction_loss = _reconstruction_loss(x, test_logits, test_dqz)

    with tf.name_scope('optimizing'):
        gan_optimizer = tf.train.AdamOptimizer(learning_rate=gan_learning_rate,
                                               beta1=train_config.adam_beta1, beta2=train_config.adam_beta2)
        reconstruction_optimizer = tf.train.AdamOptimizer(learning_rate=reconstruction_learning_rate,
                                                          beta1=train_config.adam_beta1, beta2=train_config.adam_beta2)
        gan_gradients = gan_optimizer.compute_gradients(-train_gan_loss,
                                                        tf.trainable_variables(scope="d_net[a-zA-Z0-9_]*"))
        reconstruction_gradients = reconstruction_optimizer.compute_gradients(train_reconstruction_loss,
                                                                              tf.trainable_variables(scope="(p_net|q_net)[a-zA-z_0-9]*"))
        pretrain_gradients = reconstruction_optimizer.compute_gradients(train_reconstruction_loss,
                                                                        tf.trainable_variables(scope="q_net[a-zA-Z0-9_]*"))
        if train_config.grad_clip_norm:
            for i, (grad, var) in enumerate(gan_gradients):
                if grad is not None:
                    grad = tf.clip_by_norm(
                        grad, train_config.grad_clip_norm)
                    gan_gradients[i] = (grad, var)
            for i, (grad, var) in enumerate(reconstruction_gradients):
                if grad is not None:
                    grad = tf.clip_by_norm(
                        grad, train_config.grad_clip_norm)
                    reconstruction_gradients[i] = (grad, var)
            for i, (grad, var) in enumerate(pretrain_gradients):
                if grad is not None:
                    grad = tf.clip_by_norm(
                        grad, train_config.grad_clip_norm)
                    pretrain_gradients[i] = (grad, var)
        train_op_1 = gan_optimizer.apply_gradients(gan_gradients)
        train_op_2 = reconstruction_optimizer.apply_gradients(reconstruction_gradients)
        pretrain_op = reconstruction_optimizer.apply_gradients(pretrain_gradients)

    with tf.name_scope('plotting'), \
            arg_scope([p_net], is_initializing=False):
        plot_gens = tf.sigmoid(p_net(random_z)["p_net_x"].distribution.logits)
        plot_recons = tf.sigmoid(p_net(z=q_net(x)["q_net_z"])["p_net_x"].distribution.logits)

    (x_train, y_train), (x_test, y_test) = \
        spt.datasets.load_mnist(x_shape=train_config.shape)

    train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    test_flow = bernoulli_flow(
        x_test, config.test_batch_size, sample_now=True)

    first_images = test_flow.get_arrays()[0]
    first_images = first_images[:train_config.plotting_number]

    save_images_collection(first_images*255., os.path.join(plotting_dir, 'origin.png'), grid_size=train_config.plotting_shape)

    def plotting(epoch):
        r_z = rand_z(train_config.plotting_number)
        gen = session.run(plot_gens, feed_dict={random_z:r_z})
        recon = session.run(plot_recons, feed_dict={x:first_images})
        generate_png = 'generate_' + str(epoch) + '.png'
        reconstruct_png = 'reconstruct_' + str(epoch) + '.png'
        save_images_collection(gen*255., os.path.join(plotting_dir, generate_png), grid_size=train_config.plotting_shape)
        save_images_collection(recon*255., os.path.join(plotting_dir, reconstruct_png), grid_size=train_config.plotting_shape)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow, \
            test_flow.threaded(5) as test_flow:
        spt.utils.ensure_variables_initialized()
        for [train_x] in train_flow:
            r_z = rand_z(train_config.batch_size)
            init_g_l, init_r_l = session.run([init_gan_loss, init_reconstruction_loss],feed_dict={x:train_x, random_z: r_z})
            print("init_gan_loss:", init_g_l, " init_reconstruction_loss:",
                  init_r_l)
            break

        #pretrain
        cnt=0
        for [train_x] in train_flow:
            r_z = rand_z(train_config.batch_size)
            _ = session.run([pretrain_op], feed_dict={x: train_x, random_z: r_z,
                                                      reconstruction_learning_rate: train_config.reconstruction_learning_rate})
            cnt+=1
            if(cnt==10):
                break

        for epoch in range(1,train_config.max_epoch+1):
            gan_loss = []
            reconstruction_loss = []
            start_time = time.time()
            for [train_x] in train_flow:
                r_z = rand_z(train_config.batch_size)
                train_g_l, _, dpz, dqz = session.run([train_gan_loss, train_op_1, train_dpz, train_qz],
                                           feed_dict={x:train_x, random_z:r_z, gan_learning_rate: train_config.gan_learning_rate})
                train_r_l, _ = session.run([train_reconstruction_loss, train_op_2],
                                           feed_dict={x:train_x, random_z:r_z, reconstruction_learning_rate:train_config.reconstruction_learning_rate})
                #print(r_z)
                #print(dqz)
                gan_loss.append(train_g_l)
                reconstruction_loss.append(train_r_l)
            print("epoch ", epoch, ", (time: ", time.time() - start_time, ")")
            print("train_gan_loss:", np.mean(gan_loss), " train_reconstruction_loss:",
                  np.mean(reconstruction_loss))
            if epoch % 1 ==0:
                gan_loss = []
                reconstruction_loss = []
                start_time = time.time()
                for [test_x] in test_flow:
                    r_z = rand_z(train_config.test_batch_size)
                    test_g_l, test_r_l = session.run([test_gan_loss, test_reconstruction_loss], feed_dict={x:test_x, random_z:r_z})
                    gan_loss.append(test_g_l)
                    reconstruction_loss.append(test_r_l)
                print("test_gan_loss:", np.mean(gan_loss), " test_reconstruction_loss:",
                      np.mean(reconstruction_loss))
                print("test time: ", time.time() - start_time)
                plotting(epoch)
            if epoch==30:
                train_config.gan_learning_rate/=2.
                train_config.reconstruction_learning_rate/=2.
            if epoch==50:
                train_config.gan_learning_rate/=5.
                train_config.reconstruction_learning_rate/=5.

main()
