import streamlit as st
import ast
import molvs
import rdkit
import pandas as pd
import numpy as np
from PIL import Image
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import AllChem
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from molvs.validate import Validator
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import joblib
from .DataPrepFunc import calculate_qed
from .DataPrepFunc import molecule_from_smiles
from .DataPrepFunc import smiles_to_graph
from .DataGenerator import DataGenerator
from .RelationalGraphConvLayer import RelationalGraphConvLayer
BATCH_SIZE = 20
EPOCHS =3
VAE_LR = 5e-4
BOND_DIM = 3 + 1  # Number of bond types
LATENT_DIM = 335 
class MoleculeGenerator(keras.Model):
        def __init__(self, encoder, decoder, max_len, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.property_prediction_layer = layers.Dense(1)
            self.max_len = max_len

            self.train_total_loss_tracker = keras.metrics.Mean(name="train_total_loss")
            self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        
        def train_step(self, data):
            mol_features, mol_property = data
            graph_real = mol_features
            self.batch_size = tf.shape(mol_property)[0]
            with tf.GradientTape() as tape:
                z_mean, z_log_var, property_prediction, \
                 reconstruction_adjacency, reconstruction_features = self(mol_features,
                                                                                 training=True)
                graph_generated = [reconstruction_adjacency, reconstruction_features]
                total_loss = self.calculate_loss(z_log_var,
                                                 z_mean,
                                                 mol_property,
                                                 property_prediction,
                                                 graph_real,
                                                 graph_generated,
                                                 is_train=True)

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.train_total_loss_tracker.update_state(total_loss)
            return {
                "loss": self.train_total_loss_tracker.result(),
            }

        def test_step(self, data):
            mol_features, mol_property = data
            z_mean, z_log_var, property_prediction, \
            reconstruction_adjacency, reconstruction_features = self(mol_features,
                                                                    training=False)
            total_loss = self.calculate_loss(z_log_var,
                                            z_mean,
                                            mol_property, 
                                            property_prediction,
                                            graph_real=mol_features,
                                            graph_generated=[reconstruction_adjacency, 
                                                             reconstruction_features],
                                            is_train=False)

            self.val_total_loss_tracker.update_state(total_loss)
            return {
                "loss": self.val_total_loss_tracker.result()
            }

        def calculate_loss(self,
                           z_log_var,
                           z_mean,
                           mol_property,
                           property_prediction,
                           graph_real,
                           graph_generated,
                           is_train):
            adjacency_real, features_real = graph_real
            adjacency_generated, features_generated = graph_generated
            
            adjacency_reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.categorical_crossentropy(
                                                            adjacency_real,
                                                            adjacency_generated
                                                            ),
                                                            axis=(1,2)
                        )
                )
            features_reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.categorical_crossentropy(
                                                            features_real,
                                                            features_generated
                                                            ),
                                                            axis=(1)
                        )
                )
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1)
            kl_loss = tf.reduce_mean(kl_loss)

            property_prediction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(mol_property, 
                                                    property_prediction)
            )

            if is_train:
                graph_loss = self._gradient_penalty(graph_real, graph_generated)
            else:
                graph_loss = 0

            return kl_loss + property_prediction_loss + graph_loss + adjacency_reconstruction_loss + features_reconstruction_loss

        def _gradient_penalty(self, graph_real, graph_generated):
            # Unpack graphs
            adjacency_real, features_real = graph_real
            adjacency_generated, features_generated = graph_generated

            # Generate interpolated graphs (adjacency_interp and features_interp)
            alpha = tf.random.uniform([self.batch_size])
            alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
            adjacency_interp = (adjacency_real * alpha) + (1 - alpha) * adjacency_generated
            alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
            features_interp = (features_real * alpha) + (1 - alpha) * features_generated

            # Compute the logits of interpolated graphs
            with tf.GradientTape() as tape:
                tape.watch(adjacency_interp)
                tape.watch(features_interp)
                _, _, logits, _,_ = self(
                    [adjacency_interp, features_interp], training=True
                )

            # Compute the gradients with respect to the interpolated graphs
            grads = tape.gradient(logits, [adjacency_interp, features_interp])
            # Compute the gradient penalty
            grads_adjacency_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
            grads_features_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
            return tf.reduce_mean(
                tf.reduce_mean(grads_adjacency_penalty, axis=(-2, -1))
                + tf.reduce_mean(grads_features_penalty, axis=(-1))
            )
        
        def inference(self, batch_size):
            z = tf.random.normal((batch_size, LATENT_DIM))
            reconstruction_adjacency, reconstruction_features = model.decoder.predict(z)
            # obtain one-hot encoded adjacency tensor
            adjacency = tf.argmax(reconstruction_adjacency, axis=1)
            adjacency = tf.one_hot(adjacency, depth=BOND_DIM, axis=1)
            # Remove potential self-loops from adjacency
            adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
            # obtain one-hot encoded feature tensor
            features = tf.argmax(reconstruction_features, axis=2)
            features = tf.one_hot(features, depth=ATOM_DIM, axis=2)
            return [
                graph_to_molecule([adjacency[i].numpy(), features[i].numpy()])
                for i in range(batch_size)
            ]
        
        def call(self, inputs):
            z_mean, log_var = self.encoder(inputs)
            z = Sampling()([z_mean, log_var])

            reconstruction_adjacency, reconstruction_features = self.decoder(z)

            property_prediction = self.property_prediction_layer(z_mean)

            return z_mean, log_var, property_prediction, reconstruction_adjacency, reconstruction_features