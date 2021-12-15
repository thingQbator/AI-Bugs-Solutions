Python
2.7
.18(v2
.7
.18: 8
d21aa21f2, Apr
20
2020, 13: 19:0
8) [MSC v.1500 32 bit(Intel)]
on
win32
Type
"help", "copyright", "credits" or "license()"
for more information.
    >> >
    # Commented out IPython magic to ensure Python compatibility.
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # %matplotlib inline

    from sklearn.model_selection import train_test_split

    from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import plot_model, to_categorical

    from keras.datasets.mnist import load_data


    def adam_optimizer():
        return Adam(learning_rate=0.0002, beta_1=0.5)


    def create_generator():

        """
    Create generator architecture
    """

        generator = Sequential()
        generator.add(Dense(units=256, input_dim=100))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(units=512))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(units=1024))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(units=784, activation='tanh'))

        generator.compile(loss='binary_crossentropy',
                          optimizer=adam_optimizer())
        return generator


    g = create_generator()
    g.summary()
    plot_model(g, show_shapes=True, show_layer_names=True)


    def create_discriminator():

        """
    Create discriminator architecture
    """

        discriminator = Sequential()
        discriminator.add(Dense(units=1024, input_dim=784))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(units=512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(units=256))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Dense(units=1, activation='sigmoid'))

        discriminator.compile(loss='binary_crossentropy',
                              optimizer=adam_optimizer())
        return discriminator


    d = create_discriminator()
    d.summary()
    plot_model(d, show_shapes=True, show_layer_names=True)


    def create_gan(discriminator, generator):

        # Freeze the weights
        discriminator.trainable = False

        # Initialize random noise with generator
        gan_input = Input(shape=(100,))
        x = generator(gan_input)
        gan_output = discriminator(x)

        gan = Model(inputs=gan_input, outputs=gan_output)

        gan.compile(loss='binary_crossentropy',
                    optimizer='adam')
        return gan


    gan = create_gan(d, g)
    gan.summary()
    plot_model(gan, show_shapes=True, show_layer_names=True)


    def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):

        noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(100, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i],
                       interpolation='nearest',
                       cmap='gray')
            plt.axis('off')
        plt.tight_layout()


    def training(epochs=1, batch_size=128):

        # Loading the data
        (x_train, x_test), (y_train, y_test) = load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        nb_images_train = x_train.shape[0]

        x_train = x_train.reshape(nb_images_train, 784)
        batch_count = x_train.shape[0] / batch_size

        # Creating GAN
        generator = create_generator()
        discriminator = create_discriminator()
        gan = create_gan(discriminator, generator)

        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            # tqdm()
            for _ in range(batch_size):
                # generate  random noise as an input  to  initialize the  generator
                noise = np.random.normal(0, 1, [batch_size, 100])

                # Generate fake MNIST images from noised input
                generated_images = generator.predict(noise)

                # Get a random set of  real images
                image_batch = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]

                # Construct different batches of  real and fake data
                X = np.concatenate([image_batch, generated_images])

                # Labels for generated and real data
                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9

                # Pre train discriminator on  fake and real data  before starting the gan.
                discriminator.trainable = True
                discriminator.train_on_batch(X, y_dis)

                # Tricking the noised input of the Generator as real data
                noise = np.random.normal(0, 1, [batch_size, 100])
                y_gen = np.ones(batch_size)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag
                discriminator.trainable = False

                # training  the GAN by alternating the training of the Discriminator
                # and training the chained GAN model with Discriminator’s weights freezed.
                gan.train_on_batch(noise, y_gen)

            if e == 1 or e % 20 == 0:
                plot_generated_images(e, generator)


    training(100, 128)