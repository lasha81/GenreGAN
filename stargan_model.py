import os
import time
import datetime
from glob import glob
import numpy as np
from collections import namedtuple
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from tensorflow.keras.optimizers import Adam

from stargan_module import build_generator, build_discriminator, abs_criterion, mae_criterion, discriminator_loss, \
    generator_loss, classification_loss, L1_loss
from tf2_utils import get_now_datetime, ImagePool, to_binary, load_npy_data_starGAN, save_midis, tf_shuffle_axis, concat_with_label


class StarGAN(object):

    def __init__(self, args):

        self.batch_size = args.batch_size
        self.time_step = args.time_step  # number of time steps
        self.pitch_range = args.pitch_range  # number of pitches
        self.input_c_dim = args.input_nc  # number of input image channels
        self.output_c_dim = args.output_nc  # number of output image channels
        self.lr = args.lr
        self.L1_lambda = args.L1_lambda
        self.gamma = args.gamma
        self.sigma_d = args.sigma_d
        self.dataset_A_dir = args.dataset_A_dir
        self.dataset_B_dir = args.dataset_B_dir
        self.dataset_C_dir = args.dataset_C_dir
        self.sample_dir = args.sample_dir

        self.adv_weight = args.adv_weight
        self.rec_weight = args.rec_weight
        self.cls_weight = args.cls_weight
        self.diff_weight = args.diff_weight
        self.gan_type = args.gan_type

        self.model = args.model
        self.discriminator = build_discriminator
        self.generator = build_generator
        self.number_of_domains = args.number_of_domains
        self.ld = args.ld

        self.note_threshold = args.note_threshold

        self.criterionGAN = mae_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size '
                                        'time_step '
                                        'input_nc '
                                        'output_nc '
                                        'pitch_range '
                                        'gf_dim '
                                        'df_dim '
                                        'is_training '
                                        'number_of_domains ')
        self.options = OPTIONS._make((args.batch_size,
                                      args.time_step,
                                      args.input_nc,
                                      args.output_nc,
                                      args.pitch_range,
                                      args.ngf,
                                      args.ndf,
                                      args.phase == 'train',
                                      args.number_of_domains
                                      ))
        print(self.options)

        self.now_datetime = get_now_datetime()
        self.pool = ImagePool(args.max_size)

        self._build_model(args)

        print("initialize model...")

    def _build_model(self, args):
        # Generator
        self.generator_star = self.generator(self.options,
                                             name='Generator_star')
        # Discriminator
        self.discriminator_star = self.discriminator(self.options,
                                                     name='Discriminator_star')

        """
        if self.model != 'base':
            self.discriminator_A_all = self.discriminator(self.options,
                                                          name='Discriminator_A_all')
            self.discriminator_B_all = self.discriminator(self.options,
                                                          name='Discriminator_B_all')
        """
        # Discriminator and Generator Optimizer
        self.D_optimizer = Adam(self.lr,
                                beta_1=args.beta1)
        self.G_optimizer = Adam(self.lr,
                                beta_1=args.beta1)

        """
        if self.model != 'base':
            self.DA_all_optimizer = Adam(self.lr,
                                         beta_1=args.beta1)
            self.DB_all_optimizer = Adam(self.lr,
                                         beta_1=args.beta1)
        """
        # Checkpoints
        model_name = "stargan.model"
        model_dir = "stargan_{}2{}_{}_{}_{}".format(self.dataset_A_dir,
                                                    self.dataset_B_dir,
                                                    self.now_datetime,
                                                    self.model,
                                                    self.sigma_d)
        self.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                           model_dir,
                                           model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if self.model == 'base':
            self.checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.G_optimizer,
                discriminator_optimizer=self.D_optimizer,
                generator=self.generator_star,
                discriminator=self.discriminator_star
            )
        else:
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.G_optimizer,
                                                  discriminator_optimizer=self.D_optimizer,
                                                  discriminator_all_optimizer=self.D_all_optimizer,
                                                  generator=self.generator_star,
                                                  discriminator=self.discriminator_star,
                                                  discriminator_all=self.discriminator_all)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             self.checkpoint_dir,
                                                             max_to_keep=25)

        # if self.checkpoint_manager.latest_checkpoint:
        #     self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        #     print('Latest checkpoint restored!!')

    def train(self, args):
        # Data from domain A and B, and mixed dataset for partial and full models.
        data_1 = np.array(glob('./datasets/{}/train/*.*'.format(self.dataset_A_dir)))
        data_2 = np.array(glob('./datasets/{}/train/*.*'.format(self.dataset_B_dir)))
        data_3 = np.array(glob('./datasets/{}/train/*.*'.format(self.dataset_C_dir)))

        label_1 = 0
        label_2 = 1
        label_3 = 2

        labels_1 = np.full((data_1.shape[0]), label_1)
        labels_2 = np.full((data_2.shape[0]), label_2)
        labels_3 = np.full((data_3.shape[0]), label_3)


        combined_data_1 = list(zip(data_1, labels_1))
        combined_data_2 = list(zip(data_2, labels_2))
        combined_data_3 = list(zip(data_2, labels_3))

        data_all = np.concatenate((combined_data_1, combined_data_2, combined_data_3))

        if args.continue_train:
            if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
                print(" [*] Load checkpoint succeeded!")
            else:
                print(" [!] Load checkpoint failed...")

        counter = 0
        start_time = time.time()

        # tensorboard
        g_loss_metric = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        d_loss_metric = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)

        g_adv_loss_metric = tf.keras.metrics.Mean('g_adv_loss', dtype=tf.float32)
        g_cls_loss_metric = tf.keras.metrics.Mean('g_cls_loss', dtype=tf.float32)
        g_rec_loss_metric = tf.keras.metrics.Mean('g_rec_loss', dtype=tf.float32)
        d_adv_loss_metric = tf.keras.metrics.Mean('d_adv_loss', dtype=tf.float32)
        adv_loss_metric = tf.keras.metrics.Mean('adv_loss', dtype=tf.float32)
        d_cls_loss_metric = tf.keras.metrics.Mean('d_cls_loss', dtype=tf.float32)
        number_of_notes_metric = tf.keras.metrics.Mean('number_of_notes', dtype=tf.float32)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'tb_logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # end tensorboard

        for epoch in range(args.epoch):

            # Shuffle training data
            np.random.shuffle(data_all)

            #if self.model != 'base' and data_mixed is not None:
            #    np.random.shuffle(data_mixed)

            # Get the proper number of batches
            batch_idxs = len(data_all) // self.batch_size

            # learning rate starts to decay when reaching the threshold
            self.lr = self.lr if epoch < args.epoch_step else self.lr * (args.epoch - epoch) / (
                        args.epoch - args.epoch_step)

            for idx in range(batch_idxs):

                # To feed real_data
                batch_data = data_all[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch_samples = [load_npy_data_starGAN(batch_file) for batch_file in batch_data[:,0]]
                batch_samples = np.array(batch_samples).astype(np.float32)  # batch_size * 64 * 84 * 1
                x_real = batch_samples[:, :, :, 0]
                x_real = tf.expand_dims(x_real, -1)

                # generate gaussian noise for robustness improvement
                gaussian_noise = np.abs(np.random.normal(0,
                                                         self.sigma_d,
                                                         [self.batch_size,
                                                          self.time_step,
                                                          self.pitch_range,
                                                          self.input_c_dim])).astype(np.float32)

                if self.model == 'base':

                    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:

                        labels_orig = tf.convert_to_tensor(batch_data[:, 1].astype(int))
                        labels_orig = tf.one_hot(labels_orig, depth= self.number_of_domains)
                        labels_target = tf_shuffle_axis(labels_orig, axis=1)
                        real_x_with_target_label = concat_with_label(x_real, labels_target)


                        """ Define Generator, Discriminator """

                        x_fake = self.generator_star(real_x_with_target_label, training=True)
                        fake_x_with_orig_label = concat_with_label(x_fake, labels_orig)
                        x_recon = self.generator_star(fake_x_with_orig_label, training=True)


                        real_cls = self.discriminator_star(x_real)
                        fake_cls = self.discriminator_star(x_fake)

                        """ Define Loss """
                        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
                            gp = self.gradient_penalty(real=x_real, fake=x_fake)
                        else:
                            gp = 0

                        sum_difference = tf.abs(tf.reduce_sum(x_fake)-325*self.batch_size)
                        print("sd {}".format(sum_difference.numpy()))


                        #g_adv_loss = generator_loss(loss_func=self.gan_type, fake=fake_logit)
                        #g_adv_loss = generator_loss(loss_func=self.gan_type, fake=fake_cls)
                        g_cls_loss = classification_loss(logit=fake_cls, label=labels_target)
                        g_rec_loss = L1_loss(x_real, x_recon)


                        #d_adv_loss = discriminator_loss(loss_func=self.gan_type, real=real_cls, fake=fake_cls) + gp
                        adv_loss = discriminator_loss(loss_func=self.gan_type, real=real_cls, fake=fake_cls)
                        d_cls_loss = classification_loss(logit=real_cls, label=labels_orig)

                        d_loss = -self.adv_weight * adv_loss + self.cls_weight * d_cls_loss
                        g_loss = self.adv_weight * adv_loss + self.cls_weight * g_cls_loss + self.rec_weight * g_rec_loss + self.diff_weight * sum_difference


                    # Calculate the gradients for generator and discriminator
                    generator_gradients = gen_tape.gradient(target=g_loss,
                                                            sources=self.generator_star.trainable_variables)
                    discriminator_gradients = disc_tape.gradient(target=d_loss,
                                                                 sources=self.discriminator_star.trainable_variables)

                    # Apply the gradients to the optimizer
                    self.G_optimizer.apply_gradients(zip(generator_gradients,
                                                         self.generator_star.trainable_variables))
                    self.D_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator_star.trainable_variables))

                    # tensorboard
                    g_loss_metric(g_loss)
                    d_loss_metric(d_loss)
                    #g_adv_loss_metric(g_adv_loss)
                    g_cls_loss_metric(g_cls_loss)
                    g_rec_loss_metric(g_rec_loss)
                    #d_adv_loss_metric(d_adv_loss)
                    adv_loss_metric(adv_loss)
                    d_cls_loss_metric(d_cls_loss)
                    number_of_notes_metric( len([1 for v in x_fake.numpy().flatten() if v > self.note_threshold]) / self.batch_size  )
                    # end tensorboard
                    print('=================================================================')
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f D_loss: %6.2f, G_loss: %6.2f" %
                           (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss)))

                counter += 1

                print("real {:6.2f}, fake {:6.2f}, recon {:6.2f}".format( tf.reduce_sum(x_real).numpy(), tf.reduce_sum(x_fake).numpy(), tf.reduce_sum(x_recon).numpy()))

                # save tensorboard data
                if np.mod(counter, args.tb_print_freq) == 0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('d_loss', d_loss_metric.result(), step=counter)
                        tf.summary.scalar('g_loss', g_loss_metric.result(), step=counter)
                        #tf.summary.scalar('g_adv_loss', g_adv_loss_metric.result(), step=counter)
                        tf.summary.scalar('g_cls_loss', g_cls_loss_metric.result(), step=counter)
                        tf.summary.scalar('g_rec_loss', g_rec_loss_metric.result(), step=counter)
                        #tf.summary.scalar('d_adv_loss', d_adv_loss_metric.result(), step=counter)
                        tf.summary.scalar('adv_loss', adv_loss_metric.result(), step=counter)
                        tf.summary.scalar('d_cls_loss', d_cls_loss_metric.result(), step=counter)
                        tf.summary.scalar('number_of_notes', number_of_notes_metric.result(), step=counter)


                    d_loss_metric.reset_states()
                    g_loss_metric.reset_states()
                    g_adv_loss_metric.reset_states()
                    g_cls_loss_metric.reset_states()
                    g_rec_loss_metric.reset_states()
                    d_adv_loss_metric.reset_states()
                    d_cls_loss_metric.reset_states()
                    adv_loss_metric.reset_states()
                    number_of_notes_metric.reset_states()



                # generate samples during training to track the learning process
                if np.mod(counter, args.print_freq) == 0:
                    sample_dir = os.path.join(self.sample_dir,
                                              '{}2{}_{}_{}_{}'.format(self.dataset_A_dir,
                                                                      self.dataset_B_dir,
                                                                      self.now_datetime,
                                                                      self.model,
                                                                      self.sigma_d))
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)

                    # to binary, 0 denotes note off, 1 denotes note on
                    samples = [to_binary(x_real,  self.note_threshold),
                               to_binary(x_fake,  self.note_threshold),
                               to_binary(x_recon,  self.note_threshold)]



                    self.sample_model(samples=samples,
                                      labels_orig = labels_orig,
                                      labels_target = labels_target,
                                      sample_dir=sample_dir,
                                      epoch=epoch,
                                      idx=idx)

                if np.mod(counter, args.save_freq) == 0:
                    self.checkpoint_manager.save(counter)



    def sample_model(self, samples, labels_orig, labels_target, sample_dir, epoch, idx):
        print('generating samples during learning......')

        index_orig = tf.argmax(labels_orig, axis=1)
        index_target = tf.argmax(labels_target, axis=1)

        for rec_num in range(samples[0].shape[0]):

            save_midis(tf.reshape(samples[0][rec_num],(1,64,84,1)), './{}/{:02d}_{:04d}_{:04d}_{}to{}_real.mid'.format(sample_dir, epoch, idx, rec_num, index_orig[rec_num], index_target[rec_num]))
            save_midis(tf.reshape(samples[1][rec_num],(1,64,84,1)), './{}/{:02d}_{:04d}_{:04d}_{}to{}_generated.mid'.format(sample_dir, epoch, idx, rec_num, index_orig[rec_num], index_target[rec_num], rec_num))
            save_midis(tf.reshape(samples[2][rec_num],(1,64,84,1)), './{}/{:02d}_{:04d}_{:04d}_{}to{}_reconstructed.mid'.format(sample_dir, epoch, idx, rec_num, index_orig[rec_num], index_target[rec_num], rec_num))
            np.save(('./{}/{:02d}_{:04d}_{:04d}_{}to{}_generated.npy'.format(sample_dir, epoch, idx, rec_num, index_orig[rec_num], index_target[rec_num], rec_num)),tf.reshape(samples[1][rec_num], (1, 64, 84, 1)).numpy())

        print("real {}".format(np.mean(samples[0])))
        print("fake {}".format(np.mean(samples[1])))

    def test(self, args):
        pass

    def test_famous(self, args):
        pass



    def gradient_penalty(self, real, fake, scope="discriminator"):
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random.uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random.uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake




        GP = 0

        #grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        #grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm
        interpolated = tf.Variable(interpolated)

        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(interpolated)
            logit, _ = self.discriminator_star(interpolated)

        grad = grad_tape.gradient(logit,interpolated)[0]

        grad_norm = tf.norm(grad, axis=1)  # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp' :
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP
