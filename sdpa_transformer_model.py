
import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Multihead scaled dot product attention implementation wrapper from transformer for language understanding
    example: https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb
    '''
    def __init__(self, d_model, num_heads, **kwargs):
        '''
        Arguments:
            d_model: dimension for embedding vector space.
            num_heads: number of scaled dot product attention heads
        '''
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        Arguments:
            x: Input of shape (batch_size, sea_len, d_model)
            batch_size: batch size of input x.
        Returns:
            Tensor of size (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs,  **kwargs):
        '''
        Call multihead attention to run scaled dot product on inputs split into heads.
        Arguments:
            inputs: [v, k, q, mask] value, key, query and mask inputs.
        Returns:
            output: Concatenated scaled dot product attention tensor of size (batch_size, seq_len_q, d_model)
            attention_weights: attention weights from scaled dot product attention
        '''
        v, k, q, mask= inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  #

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        '''
        Method to calculate multihead attention weights using scaled dot product.
        Arguments:
            q: query shape: (batch_size, num_heads, seq_len_q, depth)
            k: key shape: (batch_size, num_heads, seq_len_k, depth)
            v: value shape: (batch_size, num_heads, seq_len_v, depth), seq_len_k==seq_len_v is necessary.
            mask: broadcastable mask of matching attention weight shape.
        Returns:
            output: tensor of shape (batch_size, num_heads, seq_len_q, depth)
            attention_weights: tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        '''

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    '''
    Single encoder layer implementation.
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None, **kwargs):
        '''
            inputs:[x, mask]
        '''
        x, mask = inputs
        attn_output, _ = self.mha([x, x, x, mask])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        '''
        Method to implement two layer feed forward network for encoder layer.
        '''
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])


class DecoderLayer(tf.keras.layers.Layer):
    '''
    Single decoder layer implementation.
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None,**kwargs):
        '''
            inputs: [x, enc_output, look_ahead_mask, padding mask]
        '''
        x, enc_output, look_ahead_mask, padding_mask = inputs
        attn1, attn_weights_block1 = self.mha1([x, x, x, look_ahead_mask])
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2([enc_output, enc_output, out1, padding_mask])
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        '''
        Method to implement two layer feed forward network for decoder layer.
        '''
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])


class Encoder(tf.keras.layers.Layer):
    '''
    Multilayer encoder implementation.
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1, **kwargs):
        '''
            maximum_position_encoding: see pe_input definition
        '''
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None, **kwargs):
        '''
            inputs:[x, mask]
        '''
        x, mask = inputs
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i]([x, mask], training=training)

        return x  # (batch_size, input_seq_len, d_model)

    @staticmethod
    def get_angles(pos, i, d_model):
        '''
        Get angles for positional encoding in encoder layer.
        Arguments:
            pos: sequence position.
            i: feature dimension index.
        Return: Angle for sequence location pos at feature dimension index i.
        '''
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        '''
        Method to form positional encodings for encoder layer.
        Arguments:
            position: maximum number of sequence positions.
            d_model: number of feature dimensions.
        Returns:
            positional encodings
        '''
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class Decoder(tf.keras.layers.Layer):
    '''
    Multilayer decoder implementation
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None,  **kwargs):
        '''
            inputs: [x, enc_output, look_ahead_mask, padding_mask].
        '''
        x, enc_output, look_ahead_mask, padding_mask =  inputs
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i]([x, enc_output, look_ahead_mask, padding_mask], training=training)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

    @staticmethod
    def get_angles(pos, i, d_model):
        '''
        Get angles for positional encoding in encoder layer.
        Arguments:
            pos: sequence position.
            i: feature dimension index.
        Return: Angle for sequence location pos at feature dimension index i.
        '''
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        '''
        Method to form positional encodings for decoder layer.
        Arguments:
            position: maximum number of sequence positions.
            d_model: number of feature dimensions.
        Returns:
            positional encodings
        '''
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class SDPA_Transformer(tf.keras.Model):
    '''
    Scaled Dot Product Attention Transformer implementation.
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1, **kwargs):
        '''
        Initialize the transformer.
        Args:
            num_layers: number of layers for decoder and encoder
            d_model: feature dimension size for embedding space
            num_heads: number of heads for attention
            dff: number of neurons for fully-connected layers.
            input_vocab_size: vocabulary size for source language
            target_vocab_size: vocabulary size for target language
            pe_input: maximum number of positional encodings for source language
            pe_target: maximum number positional encodings for target language
            rate: drop-out rate

        '''
        super().__init__(**kwargs)

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=None, **kwargs):
        '''
        Args:
            inputs: [inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask]
            training: Boolean set True during training.
        Returns:
            Logits for output probabilities, attention weights.
        '''
        inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask=inputs
        enc_output = self.tokenizer([inp, enc_padding_mask], training=training)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder([tar, enc_output,  look_ahead_mask, dec_padding_mask], training=training)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
