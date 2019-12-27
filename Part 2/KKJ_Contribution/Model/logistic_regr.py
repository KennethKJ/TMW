from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import os
import numpy as np
import pandas as pd

# Define functions
def get_run_logdir(root_dir_fn):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M")
    full_path = os.path.join(root_dir_fn, run_id)
    os.mkdir(full_path,  mode=0o777)
    os.chdir(full_path)
    return full_path

selected_features = ['meanfun', 'IQR', 'sp.ent']
selected_features = ['IQR']

# Import data
data_frame = pd.read_csv("voices.csv")

# Extract selected features
df_x = data_frame[selected_features]
data_x = df_x.values.astype('float32')

# Apply feature crossings
do_feature_crossings = False

if do_feature_crossings:
    num_features = len(selected_features)
    notDone = True
    i1 = 0
    i2 = 0
    while i1 < num_features-1:
        i2 += 1
        data_x = np.concatenate((data_x, np.expand_dims(data_x[:, i1] * data_x[:, i2], axis=1)), axis=1)

        if i2 == num_features-1:
            i1 += 1
            i2 = i1

# Get or update the number of features
num_features = data_x.shape[1]

# Normalize faetures
for i in range(num_features):
    data_x[:, i] = (data_x[:, i]-np.mean(data_x[:, i]))/np.std(data_x[:, i])

# Extract labels as ones and zeros
data_y = pd.Categorical(data_frame.values[:, -1]).codes

# Get number of data points
num_data = data_frame.shape[0]

# Randomize data
np.random.seed(seed=0)  # fix seed to get same sequence
idx = np.random.permutation(num_data)

# Define test, eval, train proportions
proportion_train = 0.7
proportion_eval = 0.2
proportion_test = 1 - proportion_train - proportion_eval

# Extract train, eval, and test datasets
idx_train = idx[0: int(proportion_train*num_data)]
X_train = data_x[idx_train, :]
Y_train = data_y[idx_train]

idx_eval = idx[int(proportion_train*num_data): len(idx_train) + int(proportion_eval*num_data)]
X_eval = data_x[idx_eval, :]
Y_eval = data_y[idx_eval]

idx_test = idx[len(idx_train) + len(idx_eval):]
X_test = data_x[idx_test, :]
Y_test = data_y[idx_test]

# Define model
num_units = 1
dim_input = num_features
learning_rate = 0.001
num_epochs = 70
batch_size = 32
root_dir = 'C:\\Users\\alert\\Google Drive\\TMW Working dir\\Keras\\'

model = Sequential()
model.add(Dense(units=num_units, activation='sigmoid', input_dim=dim_input))

optim = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy',
              optimizer=optim,
              metrics=['accuracy'])

# Print summary of model
model.summary()

# Set main training directory
run_logdir = get_run_logdir(root_dir)

# Define callbacks
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint("saved_model.h5", save_best_only=True)

# Run the fitting
history = model.fit(X_train, Y_train,
                    epochs=num_epochs,
                    steps_per_epoch=round(X_train.shape[0]/batch_size),
                    batch_size=batch_size,
                    validation_data =(X_eval, Y_eval),
                    validation_steps=100,
                    callbacks=[checkpoint_cb, tensorboard_cb]
                    )

# Save model at end
runID = "Model_NU" + str(num_units) + "_LR=" + "_BZ=" + str(batch_size) + "_NE=" + str(num_epochs) + "_DI=" + str(dim_input)
model.save(run_logdir + "\\" + runID + '.h5')

doTest = True
if doTest:
    classes = model.predict(X_test, batch_size=128)
    classes = classes > 0.5
    classes = np.squeeze(classes)

    truth = Y_test > 0.5

    reality = classes == truth
    hit = np.round(np.sum(reality)/len(truth)*100)

    print('Test set accuracy = ' + str(hit))


print("Done!")