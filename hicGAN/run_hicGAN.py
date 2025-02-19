# coding: utf-8
import os, time, pickle, random, time, sys, math
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow.compat.v1 as tf
import tensorlayer as tl
from tensorlayer.layers import *
import matplotlib.pyplot as plt
import hickle as hkl
from skimage.measure import compare_mse
from skimage.measure import compare_ssim

tf.disable_v2_behavior()

usage='''
Usage: python run_hicGAN.py  <GPU_ID> <checkpoint> <graph> <CELL>
-- a program for running hicGAN model
OPTIONS:
    <GPU_ID> -- GPU ID
    <checkpoint> -- path to save model weights at different training epoch
    <graph> -- path to save event file for TensorBoard visualization
    <CELL> -- selected cell type
'''
if len(sys.argv)!=5:
    print(usage)
    sys.exit(1)
#GPU setting and Global parameters
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
#if tf.test.gpu_device_name():
#  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#else:
#   print("Please install GPU version of TF")
checkpoint = sys.argv[2]
graph_dir = sys.argv[3]
tl.global_flag['mode']='hicgan'
tl.files.exists_or_mkdir(checkpoint)
tl.files.exists_or_mkdir(graph_dir)
batch_size = 128
lr_init = 1e-4
cell_type = sys.argv[4]
beta1 = 0.9
n_epoch_init = 1
n_epoch = 500
lr_decay = 0.1
decay_every = int(n_epoch / 2)
ni = int(np.sqrt(batch_size))


def calculate_psnr(mat1,mat2):
    data_range=np.max(mat1)-np.min(mat1)
    err=compare_mse(mat1,mat2)
    return 10 * np.log10((data_range ** 2) / err)

def calculate_ssim(mat1,mat2):
    data_range=np.max(mat1)-np.min(mat1)
    return compare_ssim(mat1,mat2,data_range=data_range)

#load data

lr_mats_train_full,hr_mats_train_full = (np.load('../lowres.npy'), np.load('../highres.npy')) #hkl.load('data/%s/train_data.hkl'%cell_type)

lr_mats_train = lr_mats_train_full[:int(0.95*len(lr_mats_train_full))]
hr_mats_train = hr_mats_train_full[:int(0.95*len(hr_mats_train_full))]

lr_mats_valid = lr_mats_train_full[int(0.95*len(lr_mats_train_full)):]
hr_mats_valid = hr_mats_train_full[int(0.95*len(hr_mats_train_full)):]



# Model implementation
    
def hicGAN_g(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("hicGAN_g", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n
        # B residual blocks
        for i in range(5):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end. output shape: (None,w,h,64)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n128s1/1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n
    

def hicGAN_d(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("hicGAN_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c1')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b1')
        #output shape: (None,w/2,h/2,64)
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b2')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c3')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b3')
        #output shape: (None,w/4,h/4,64)
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c4')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b4')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c5')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b5')
        #output shape: (None,w/8,h/8,256)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        #n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')
        #output shape: (None,w/16,h/16,512)
        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=512, act=lrelu, name='d512')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits



# In[124]:


t_image = tf.placeholder('float32', [None, 40, 40, 1], name='input_to_generator')
t_target_image = tf.placeholder('float32', [None, 40, 40, 1], name='t_target_hic_image')

net_g = hicGAN_g(t_image, is_train=True, reuse=False)
net_d, logits_real = hicGAN_d(t_target_image, is_train=True, reuse=False)
_, logits_fake = hicGAN_d(net_g.outputs, is_train=True, reuse=True)

net_g_test = hicGAN_g(t_image, is_train=False, reuse=True)
d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
d_loss = d_loss1 + d_loss2
g_gan_loss = 1e-1 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
#g_loss = mse_loss + g_gan_loss
g_loss = g_gan_loss
#losses are all based on a batch data
g_vars = tl.layers.get_variables_with_name('hicGAN_g', True, True)
d_vars = tl.layers.get_variables_with_name('hicGAN_d', True, True)

with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_init, trainable=False)



g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

#summary variables
tf.summary.scalar("d_loss1", d_loss1)
tf.summary.scalar("d_loss2", d_loss2)
tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("mse_loss", mse_loss)
tf.summary.scalar("g_gan_loss", g_gan_loss)
tf.summary.scalar("g_combine_loss", 1e-1*g_gan_loss+mse_loss)
merged_summary = tf.summary.merge_all()

#Model pretraining G
config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction=0.9
sess = tf.Session(config=config)
tl.layers.initialize_global_variables(sess)

#record variables for TensorBoard visualization
summary_writer=tf.summary.FileWriter('%s'%graph_dir,graph=tf.get_default_graph())

# sess.run(tf.assign(lr_v, lr_init))
# print(" ** fixed learning rate: %f (for init G)" % lr_init)
# f_out = open('%s/pre_train.log'%log_dir,'w')
# for epoch in range(0, n_epoch_init + 1):
#     epoch_time = time.time()
#     total_mse_loss, n_iter = 0, 0
#     for idx in range(0, len(hr_mats_train_scaled)-batch_size, batch_size):
#         step_time = time.time()
#         b_imgs_input = lr_mats_train_scaled[idx:idx + batch_size]
#         b_imgs_target = hr_mats_train_scaled[idx:idx + batch_size]
#         #b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
#         #b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
#         ## update G
#         errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_input, t_target_image: b_imgs_target})
#         print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
#         total_mse_loss += errM
#         n_iter += 1
#     log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f\n" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
#     print(log)
#     f_out.write(log)
# f_out.close()
#out = sess.run(net_g_test.outputs, {t_image: test_sample})
#print("[*] save images")
#tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
## save model
#if (epoch != 0) and (epoch % 10 == 0):
#tl.files.save_npz(net_g.all_params, name=checkpoint + '/g_{}_init_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)


# In[128]:


###========================= train GAN (hicGAN) =========================###
wait=0
patience=20
best_mse_val = np.inf
for epoch in range(0, n_epoch + 1):
    ## update learning rate
    if epoch != 0 and (epoch % decay_every == 0):
        #new_lr_decay = lr_decay**(epoch // decay_every)
        new_lr_decay=1
        sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
        log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
        print(log, flush=True)
    elif epoch == 0:
        sess.run(tf.assign(lr_v, lr_init))
        log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
        print(log, flush=True)

    epoch_time = time.time()
    total_d_loss, total_g_loss, n_iter = 0, 0, 0

    for idx in range(0, len(hr_mats_train)-batch_size, batch_size):
        step_time = time.time()
        b_imgs_input = lr_mats_train[idx:idx + batch_size]
        b_imgs_target = hr_mats_train[idx:idx + batch_size]
        ## update D
        errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_input, t_target_image: b_imgs_target})
        ## update G
        errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim], {t_image: b_imgs_input, t_target_image: b_imgs_target})
        print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f  adv: %.6f)" %
              (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA), flush=True)
        total_d_loss += errD
        total_g_loss += errG
        n_iter += 1
    #validation
    print(hr_mats_valid.shape)
    hr_mats_pre = np.zeros((hr_mats_valid.shape))
    for i in range(int(hr_mats_pre.shape[0]/batch_size)):
        hr_mats_pre[batch_size*i:batch_size*(i+1)] = sess.run(net_g_test.outputs, {t_image: lr_mats_valid[batch_size*i:batch_size*(i+1)]})
    hr_mats_pre[batch_size*(i+1):] = sess.run(net_g_test.outputs, {t_image: lr_mats_valid[batch_size*(i+1):]})
    mse_val=np.median(list(map(compare_mse,hr_mats_pre[:,:,:,0],hr_mats_valid[:,:,:,0])))
    if mse_val < best_mse_val:
        wait=0
        best_mse_val = mse_val
        #save the model with minimal MSE in validation samples
        tl.files.save_npz(net_g.all_params, name=checkpoint + '/g_{}_best.npz'.format(tl.global_flag['mode']), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint + '/d_{}_best.npz'.format(tl.global_flag['mode']), sess=sess)
    else:
        wait+=1
        if wait >= patience:
            print("Early stopping! The validation median mse is %.6f\n"%best_mse_val, flush=True)
            #sys.exit() 

    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f valid_mse:%.8f\n" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,total_g_loss / n_iter,mse_val)
    print(log)
    #record variables for TensorBoard visualization
    summary=sess.run(merged_summary,{t_image: b_imgs_input, t_target_image: b_imgs_target})
    summary_writer.add_summary(summary, epoch)
    
    

    ## quick evaluation on test sample
#     if (epoch != 0) and (epoch % 5 == 0):
#         out = sess.run(net_g_test.outputs, {t_image: test_sample})  #; print('gen sub-image:', out.shape, out.min(), out.max())
#         print("[*] save images")
#         tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

    ## save model every 5 epochs
    if (epoch <=5) or ((epoch != 0) and (epoch % 5 == 0)):
        tl.files.save_npz(net_g.all_params, name=checkpoint + '/g_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint + '/d_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)










