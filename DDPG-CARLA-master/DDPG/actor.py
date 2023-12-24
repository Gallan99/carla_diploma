import keras.backend as keras_backend
import tensorflow as tf
from keras.initializers import normal
from keras.layers import Dense, Input
from keras.models import Model
from carla_config import hidden_units, image_network


class ActorNetwork:
    def __init__(self, tf_session, state_size, action_size=1, tau=0.001, lr=0.0001):
        self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr

        keras_backend.set_session(tf_session)
        # make the Q-network
        self.model, self.model_states = self.generate_model()
        # return a list of all trainable weight variables of the layers.
        # These are the weights that get updated by the optimizer during training.
        model_weights = self.model.trainable_weights
        # make the target network for the future Q
        self.target_model, _ = self.generate_model()

        # Generate tensors to hold the gradients for Policy Gradient update
        # A placeholder is a variable in Tensorflow to which data will be assigned sometime later on
        # return A Tensor that can be used to feed a value but cannot be evaluated directly.
        self.action_gradients = tf.placeholder(tf.float32, [None, action_size])
        # model.output is a Tensor or list of Tensors that need to be differentiated. model_weights is a Tensor or
        # model_weights is a Tensor or list of Tensors which is used for differentiation.
        # action_gradients is a Tensor or list of Tensors that is used to compute gradients for y.
        # Returns: A list of Tensor of length len(model_weights) where each tensor is the sum(
        # dy/dx) for y in model.output and for x in model_weights.
        self.parameter_gradients = tf.gradients(self.model.output, model_weights, -self.action_gradients)
        # The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed
        # iterator is paired together, and then the second item in each passed iterator are paired together etc.
        # for its weight their gradients parameter
        self.gradients = zip(self.parameter_gradients, model_weights)
        # minimize the gradients via adam algorithm and change the weights
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.gradients)
        self.tf_session.run(tf.global_variables_initializer())

    def train(self, states, action_gradients):
        self.tf_session.run(
            self.optimize,
            feed_dict={
                self.model_states: states,
                self.action_gradients: action_gradients,
            },
        )
    # train the target network via the weights from the main network
    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(target_weights)

    def generate_model(self):
        input_layer = Input(shape=[self.state_size])
        # 100 neurons
        h0 = Dense(hidden_units[0], activation="relu")(input_layer)
        # 400 neurons
        h1 = Dense(hidden_units[1], activation="relu")(h0)
        # 2 neurons (maybe one for steer and the other for throttle/brake)
        output_layer = Dense(2, activation="tanh")(h1)
        model = Model(input=input_layer, output=output_layer)
        tf.keras.utils.plot_model(model,
                                  to_file=image_network + 'actor_model_WP_Carla.png',
                                  show_shapes=True,
                                  show_layer_names=True, rankdir='TB')

        # input-->h0-->h1-->output

        return model, input_layer
