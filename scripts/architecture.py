from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def DeepSetsAtt(
        input_particles,
        input_time,
        input_conditional,
        input_mask,
        num_feat,
        num_heads=4,
        num_transformer = 4,
        projection_dim=128,
        num_embed = 64,
):

    time_embedding = FourierProjection(input_time,num_embed)
    jet_conditional = get_encoding(input_conditional,projection_dim)
    conditional_embeding = layers.Concatenate(-1)([time_embedding,jet_conditional])
    conditional = layers.Dense(2*projection_dim,activation="gelu")(conditional_embeding)
    conditional = layers.Reshape((1,-1))(conditional)
    conditional = tf.tile(conditional,(1,tf.shape(input_particles)[1],1))
    scale,shift = tf.split(conditional,2,-1)
        
    masked_inputs = layers.Masking(mask_value=0.0,name='Mask')(input_particles)
    masked_features = layers.Dense(projection_dim,activation="gelu")(masked_inputs)
    masked_features = masked_features*(1.0 + scale) + shift
    

    encoded_patches = masked_features    
    mask_matrix = tf.matmul(input_mask,input_mask,transpose_b=True)
    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.GroupNormalization(groups=1)(encoded_patches)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads
        )(x1, x1, attention_mask=tf.cast(mask_matrix,tf.bool))
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.
        x3 = layers.GroupNormalization(groups=1)(x2)
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        

    representation = layers.GroupNormalization(groups=1)(encoded_patches + masked_features)
    representation = layers.Dense(2*projection_dim,activation="gelu")(representation)
    outputs = layers.Dense(num_feat,activation=None)(representation)*input_mask
    
    return  outputs


def Resnet(inputs,
           inputs_time,
           num_jet,               
           num_layer = 3,
           projection_dim=128,
           dropout=0.0,
           ):
    
    def resnet_dense(input_layer,hidden_size,nlayers=2):
        x = input_layer
        residual = layers.Dense(hidden_size)(x)
        for _ in range(nlayers):
            x = layers.Dense(hidden_size,activation='swish')(x)
            x = layers.Dropout(dropout)(x)
        return residual + x

    time = FourierProjection(inputs_time,projection_dim)
    cond_token = layers.Dense(2*projection_dim,activation='gelu')(time)
    scale,shift = tf.split(cond_token,2,-1)
        
    layer = layers.Dense(projection_dim,activation='swish')(inputs)
    #layer = layers.LayerNormalization(epsilon=1e-6)(layer)
    layer = layer*(1.0+scale) + shift
    
    for _ in range(num_layer-1):
        layer = layers.LayerNormalization(epsilon=1e-6)(layer)
        layer =  resnet_dense(layer,projection_dim)

    layer = layers.LayerNormalization(epsilon=1e-6)(layer)
    outputs = layers.Dense(num_jet,kernel_initializer="zeros")(layer)
    
    return outputs


def FourierProjection(x,projection_dim):
    half_dim = projection_dim // 2
    emb = tf.math.log(10000.0) / (half_dim - 1)
    emb = tf.cast(emb,tf.float32)
    freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        
        
    angle = x*freq*1000.0
    embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)*x
    embedding = layers.Dense(2*projection_dim,activation="swish",use_bias=False)(embedding)
    embedding = layers.Dense(projection_dim,activation="swish",use_bias=False)(embedding)
    
    return embedding

def get_encoding(x,projection_dim,use_bias=True):
    x = layers.Dense(2*projection_dim,use_bias=use_bias,activation='gelu')(x)
    x = layers.Dense(projection_dim,use_bias=use_bias,activation='gelu')(x)
    return x


def DeepSetsAttClass(
        num_feat,
        num_heads=4,
        num_transformer = 4,
        projection_dim=32,
):


    inputs = Input((None,num_feat))
    masked_inputs = layers.Masking(mask_value=0.0,name='Mask')(inputs)


    masked_features = masked_inputs        
    masked_features = TimeDistributed(Dense(projection_dim,activation=None))(masked_features)
    
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event

    tdd = TimeDistributed(Dense(projection_dim,activation=None))(masked_features)
    tdd = TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = TimeDistributed(Dense(projection_dim))(tdd)


    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #x1 =encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads,
            dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        
        # Layer normalization 2.

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)                
        x3 = layers.Dense(4*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)
        
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    pooled = layers.GlobalAvgPool1D()(representation)
    representation = Dense(2*projection_dim,activation=None)(pooled)
    representation = layers.Dropout(0.1)(representation)
    representation = layers.LeakyReLU(alpha=0.01)(representation)
        
    outputs = Dense(1,activation='sigmoid')(representation)
    
    return  inputs, outputs
