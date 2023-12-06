import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""A semente, muitas vezes referida como "seed" em inglês,
é um número inicial usado para inicializar a geração de números pseudoaleatórios.
Ao definir uma semente, você torna os resultados da geração de números pseudoaleatórios determinísticos,
o que significa que, se você usar a mesma semente, obterá exatamente a mesma sequência de
números pseudoaleatórios toda vez que executar o seu código."""

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

"""
This network learns two functions:

Actor: This takes as input the state of our environment and
returns a probability value for each action in its action space.

Critic: This takes as input the state of our environment and
returns an estimate of total rewards in the future.
"""

num_inputs = 4
num_hidden = 128
num_actions = 2

inputs = layers.Input(shape=(num_inputs,))  # Neuronios de entrada
common = layers.Dense(num_hidden, activation="relu")(
    inputs
)  # Neuronios ocultos, conectada a camada inputs
action = layers.Dense(num_actions, activation="softmax")(
    common
)  # Neuronios de saida, conectada a camada common
critic = layers.Dense(1)(common)

try:
    model = keras.models.load_model("models")
    print("Model loaded")
except:
    model = keras.Model(inputs=inputs, outputs=[action, critic])
    print("Creating new model...")

# Recompila o modelo
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=[None, "mean_squared_error"])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()  # Usado para calcular a perda
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    # Reseta o ambiente
    state = env.reset()
    episode_reward = 0

    with tf.GradientTape() as tape:  # Grava as operações executadas dentro do contexto
        for timestep in range(1, max_steps_per_episode):
            # env.render(mode="human")
            # Renderiza o ambiente

            # Pega a ação e a probabilidade de cada ação
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # update running reward to check ocndition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic

        returns = []
        discount_sum = 0

        for r in rewards_history[::-1]:
            discount_sum = r + gamma * discount_sum
            returns.insert(0, discount_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)

        optimizer.apply_gradients(
            zip(grads, model.trainable_variables)
        )  # aqui ele realmente aprende

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count), flush=True)

    if running_reward > 150:  # Condition to consider the task solved
        print(f"Solved at episode {episode_count}!", flush=True)
        model.save("models")
        break

env.close()
