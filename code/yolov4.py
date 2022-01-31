#Yolov4,  YOLOv3, YOLOv3_tiny definitions
import os
import tensorflow as tf
import numpy as np

class YOLOv4:
    def __init__(self,
                 class_num,
                 anchors,
                 iou_threshold,
                 score_threshold,
                 anchors_mask,
                 training=False,
                 batch_norm_decay=0.9,
                 weight_decay=5e-4,
                 reuse=False):
        self.class_num = class_num
        self.anchors = anchors
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.anchors_mask = anchors_mask
        self.training = training
        self.batch_norm_decay = batch_norm_decay
        self.weight_decay = weight_decay
        self.reuse = reuse

    def __call__(self, inputs, name=None):
        with tf.variable_scope(name, reuse=self.reuse):
            inputs = self._conv2d_fixed_padding(inputs, filters=32, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block1')
            inputs = self._conv2d_fixed_padding(inputs, filters=64, kernel_size=3, strides=2)
            inputs = self._batch_norm_relu(inputs, name='block2')
            inputs = self._conv2d_fixed_padding(inputs, filters=128, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block3')
            inputs = self._conv2d_fixed_padding(inputs, filters=64, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block4')
            inputs = self._conv2d_fixed_padding(inputs, filters=128, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block5')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=3, strides=2)
            inputs = self._batch_norm_relu(inputs, name='block6')
            inputs = self._conv2d_fixed_padding(inputs, filters=128, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block7')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block8')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=2)
            inputs = self._batch_norm_relu(inputs, name='block9')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block10')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block11')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block12')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block13')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block14')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block15')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block16')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block17')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block18')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=2)
            inputs = self._batch_norm_relu(inputs, name='block19')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block20')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block21')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block22')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block23')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=2)
            inputs = self._batch_norm_relu(inputs, name='block24')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block25')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block26')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block27')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block28')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block29')  
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block30')

            # Top-down layers
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block31')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=2)
            inputs = self._batch_norm_relu(inputs, name='block32')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block33')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block34')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block35')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block36')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block37')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block38')
            inputs = self._conv2d_fixed_padding(inputs, filters=256, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block39')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block40')

            # Middle flow
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=2)
            inputs = self._batch_norm_relu(inputs, name='block41')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block42')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block43')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block44')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block45')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block46')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block47')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block48')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block49')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block50')

            # Bottom-up layers
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block51')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block52')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block53')
            inputs = self._conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block54')
            inputs = self._conv2d_fixed_padding(inputs, filters=512, kernel_size=1, strides=1)
            inputs = self._batch_norm_relu(inputs, name='block55')

            # Classification layer
            inputs = self._conv2d_fixed_padding(inputs, filters=self.num_classes, kernel_size=1, strides=1)
            inputs = tf.identity(inputs, 'final_layer')
            
            return inputs

    def _conv2d_fixed_padding(self, inputs, filters, kernel_size, strides):
        """
        A fixed-padding implementation of convolution.
        """
        if strides > 1:
            inputs = self._fixed_padding(inputs, kernel_size)
        inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides,
                                  kernel_initializer=tf.variance_scaling_initializer(),
                                  padding=('SAME' if strides == 1 else 'VALID'),
                                  use_bias=False,
                                  name='conv2d')
        return inputs

    def _fixed_padding(self, inputs, kernel_size):
        """
        Pads the input along the spatial dimensions independently of input size.
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return inputs

    def _batch_norm_relu(self, inputs, name):
        """
        Performs a batch normalization followed by a ReLU.
        """
        inputs = tf.layers.batch_normalization(inputs, training=self.is_training, name=name)
        inputs = tf.nn.relu(inputs)
        return inputs

    def _build_model(self):
        """
        Builds the model.
        """
        inputs = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], 'inputs')
        self.inputs = inputs
        self.is_training = tf.placeholder_with_default(False, [], 'is_training')
        self.logits = self._build_model_graph(inputs)
        self.predictions = tf.argmax(self.logits, axis=-1, output_type=tf.int32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.labels), tf.float32))

    def _build_train_op(self):
        """
        Builds the training operation.
        """
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step,
                                                        self.decay_steps, self.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_summary_op(self):
        """
        Builds the summary operation.
        """
        self.summary_op = tf.summary.merge_all()

    def _build_summary_writer(self):
        """
        Builds the summary writer.
        """
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def _build_saver(self):
        """
        Builds the saver.
        """
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    
    def train(self, train_dataset, val_dataset, num_epochs, batch_size, learning_rate, decay_steps, decay_rate,
                initial_learning_rate, max_to_keep, log_dir, restore_checkpoint, save_checkpoint_steps,
                save_summary_steps, checkpoint_dir, val_steps, val_batch_size):
            """
            Trains the model.
            """
            self._build_model()
            self._build_train_op()
            self._build_summary_op()
            self._build_summary_writer()
            self._build_saver()
            self.sess.run(tf.global_variables_initializer())
    
            # Restore the checkpoint if it exists
            if restore_checkpoint:
                checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                if checkpoint:
                    self.saver.restore(self.sess, checkpoint)
    
            # Start the queue runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
    
            # Iterate over the training steps
            for epoch in range(num_epochs):
                for step in range(save_checkpoint_steps):
                    # Get the next batch
                    images, labels = self.sess.run([train_dataset.images, train_dataset.labels])
    
                    # Run the training step
                    _, loss, accuracy, learning_rate_val, global_step_val = self.sess.run(
                        [self.train_op, self.loss, self.accuracy, self.learning_rate, self.global_step],
                        feed_dict={self.inputs: images, self.labels: labels, self.is_training: True})
    
                    # Save the summary
                    if step % save_summary_steps == 0:
                        summary = self.sess.run(self.summary_op, feed_dict={self.inputs: images, self.labels: labels,
                                                                            self.is_training: False})
                        self.summary_writer.add_summary(summary, global_step=global_step_val)
    
                    # Save the checkpoint
                    if step % save_checkpoint_steps == 0:
                        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=global_step_val)

                # Run the validation steps
                for step in range(val_steps):
                    images, labels = self.sess.run([val_dataset.images, val_dataset.labels])
                    loss, accuracy = self.sess.run([self.loss, self.accuracy],
                                                   feed_dict={self.inputs: images, self.labels: labels,
                                                              self.is_training: False})
                    print('Epoch: {}, Step: {}, Validation Loss: {:.5f}, Validation Accuracy: {:.5f}'.format(
                        epoch + 1, step + 1, loss, accuracy))

                # Save the checkpoint
                self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=global_step_val)

            # Stop the threads
            coord.request_stop()
            coord.join(threads)

    def predict(self, dataset, batch_size):
        """
        Predicts the labels of the given dataset.
        """
        # Start the queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        # Iterate over the images
        predictions = []
        for step in range(dataset.num_examples // batch_size):
            images = self.sess.run(dataset.images)
            predictions.extend(self.sess.run(self.predictions, feed_dict={self.inputs: images, self.is_training: False}))

        # Stop the threads
        coord.request_stop()
        coord.join(threads)

        return predictions  # list of labels

    def evaluate(self, dataset, batch_size):
        """
        Evaluates the model on the given dataset.
        """
        # Start the queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        # Iterate over the images
        accuracies = []
        for step in range(dataset.num_examples // batch_size):
            images, labels = self.sess.run([dataset.images, dataset.labels])
            accuracy = self.sess.run(self.accuracy, feed_dict={self.inputs: images, self.labels: labels, self.is_training: False})
            accuracies.append(accuracy)

        # Stop the threads
        coord.request_stop()
        coord.join(threads)

        return np.mean(accuracies)

    def save(self, checkpoint_dir, global_step):
        """
        Saves the model.
        """
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=global_step)

    def load(self, checkpoint_dir):
        """
        Loads the model.
        """
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint:
            self.saver.restore(self.sess, checkpoint)

    def close(self):
        """
        Closes the session.
        """
        self.sess.close()

    def __del__(self):
        self.close()


