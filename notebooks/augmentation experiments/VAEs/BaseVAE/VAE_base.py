#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
print(sys.executable)
print(sys.path)


import os , warnings , sys
warnings.filterwarnings("ignore")
from abc import ABC 
import tensorflow as tf
import numpy as np 
import keras

import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer , Dense, Input, Flatten, Reshape
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping


# In[ ]:


class Sampling(Layer):
    def call(self,inputs):
        # inputs = The encoder outputs two things: mean and log-variance of the latent distribution.
        # The sampling layer then produces the actual latent vector z that goes into the decoder.
        z_mean,z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim =tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch,dim))
        return z_mean+tf.exp(0.5 * z_log_var) * epsilon 


# In[ ]:


class BaseVariationalAutoencoder(Model):
    model_name = "BaseVAE"
    def __init__(self,seq_len,feat_dim,latent_dim,reconstruction_wt=3.0,batch_size=16,**kwargs):
        super().__init__(**kwargs)
        self.seq_len=seq_len
        self.feat_dim= feat_dim
        self.latent_dim = latent_dim 
        self.reconstruction_wt= reconstruction_wt
        self.batch_size=batch_size

        self.total_loss_tracker= Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        
        self.encoder= self._get_encoder()
        self.decoder= self._get_decoder()
        
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def _get_encoder(self):
        """ default vanilla encoder"""
        inputs= Input(shape=(self.seq_len,self.feat_dim))
        x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(inputs)
        x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
        self._shape_before_flattening = keras.backend.int_shape(x)[1:] 
        x= Flatten()(x)
        
        z_mean=Dense(self.latent_dim,name="z_mean")(x)
        z_log_var=Dense(self.latent_dim,name="z_log_var")(x)
        z = Sampling()([z_mean,z_log_var])
        return Model(inputs,[z_mean,z_log_var,z],name="encoder")
    
    def _get_decoder(self):
        latent_inputs=Input(shape=(self.latent_dim,))

        # Calculate nodes needed to reshape back to Conv1D space
        nodes_needed = np.prod(self._shape_before_flattening)
        x = layers.Dense(nodes_needed, activation="relu")(latent_inputs)
        x = layers.Reshape(self._shape_before_flattening)(x)
       
        x=layers.Conv1DTranspose(64,3,activation="relu",strides=2,padding="same")(x)
        x=layers.Conv1DTranspose(32,3,activation='relu',strides=2,padding="same")(x)

        outputs= layers.Conv1DTranspose(filters= self.feat_dim,kernel_size=3,activation="sigmoid",padding="same")(x)
        outputs = outputs[:, :self.seq_len, :]
        return Model(latent_inputs,outputs,name="decoder")
    

    def fit_on_data(self,train_data,max_epochs=500,verbose=0):
        
        early_stopping=EarlyStopping(monitor="total_loss",min_delta=1e-2,patience=50,mode="min")
        reduce_lr = ReduceLROnPlateau(
            monitor="total_loss",factor=0.5,patience=30, mode="min"
        )
        self.fit(
            train_data,
            batch_size=self.batch_size,
            epochs=max_epochs,
            verbose=verbose,
            callbacks=[early_stopping,reduce_lr]
            
        )
    
    def call(self,x):
        # the decoder should return something with shape [batch_size, seq_len, feat_dim] (for time-series VAEs) 
        # or [batch_size, feat_dim] (for simpler cases).
        _,_, z = self.encoder(x)

        return self.decoder(z)
    
    def get_num_trainable_variables(self):
        trainable_params= int(
            np.sum([np.prod(v.get_shape())for v in self.trainable_weights]))
        
        non_trainable_params = int(
            np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights]))

        total_params = trainable_params + non_trainable_params
        return trainable_params , non_trainable_params , total_params
    
    def get_prior_samples(self, num_samples):
        #generate synthetic samples from the prior distribution
        Z = np.random.randn(num_samples,self.latent_dim)
        samples = self.decoder.predict(Z,verbose=0)
        return samples
    
    def get_prior_samples_givenZ(self,Z):
        samples=self.decoder.predict(Z)
        return samples 
    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, data):
        if isinstance(data, tuple): 
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean,z_log_var,z=self.encoder(data)
            reconstruction = self.decoder(z)
            mse_per_point = tf.square(data - reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(mse_per_point, axis=[1, 2]))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = (self.reconstruction_wt * reconstruction_loss) + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        mse_per_point = tf.square(data - reconstruction)

        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(mse_per_point, axis=[1, 2]))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = (self.reconstruction_wt * reconstruction_loss) + kl_loss

    
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(
            encoder_wts, os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        joblib.dump(
            decoder_wts, os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

    def load_weights(self, model_dir):
        encoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        decoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

