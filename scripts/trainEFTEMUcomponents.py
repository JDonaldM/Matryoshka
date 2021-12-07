import numpy as np
from tensorflow.python.keras.utils.generic_utils import default
import matryoshka.training_funcs as MatTrain
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputX", help="Directory with feature files.")
parser.add_argument("--inputY", help="Directory with target function files.")
parser.add_argument("--cache", help="Path to save outputs.")
parser.add_argument("--new_split", help='Use a new train test split? 0 for no, 1 for yes', 
                    default=0)
parser.add_argument("--archP110", help="Architecture for P110 emulator. pass as a string i.e. '200 200'. This specifies two hidden layers with 200 nodes each.", 
                    default="200 200")
parser.add_argument("--archP112", help='Architecture for P112 emulator.', 
                    default="200 200")
parser.add_argument("--archPloop0", help='Architecture for Ploop0 emulator.', 
                    default="600 600")
parser.add_argument("--archPloop2", help='Architecture for Ploop2 emulator.', 
                    default="600 600")
parser.add_argument("--archPct0", help='Architecture for Pct0 emulator.', 
                    default="200 200")
parser.add_argument("--archPct2", help='Architecture for Pct2 emulator.', 
                    default="200 200")
parser.add_argument("--verbose", help='Verbose for tensorflow.', default=0)
parser.add_argument("--to_train", help="Componenets to train. Pass as a string i.e. 'Ploop Pct'. This will only train the Ploop and Pct components.",
                    default="P11 Ploop Pct")
args = parser.parse_args()

inputX_path = args.inputX
inputY_path = args.inputY
cache_path = args.cache
new_split = bool(args.new_split)
arch_dict = {'P11':[[int(i) for i in args.archP110.split(" ")], [int(i) for i in args.archP112.split(" ")]], 
             'Ploop':[[int(i) for i in args.archPloop0.split(" ")], [int(i) for i in args.archPloop2.split(" ")]],
             'Pct':[[int(i) for i in args.archPct0.split(" ")], [int(i) for i in args.archPct2.split(" ")]]}

print("Loading features...")
cosmos = []
for file in sorted(os.listdir(inputX_path)):
    print(file)
    cosmos.append(np.load(inputX_path+file))
cosmos = np.vstack(cosmos)
Nsamp = cosmos.shape[0]
print("Done.")

print("Splitting into train and test sets...")
if os.path.isfile(cache_path+"split/train.npy") and not new_split:
    print("Loaded old split...")
    train_id = np.load(cache_path+"split/train.npy")
    test_id = np.load(cache_path+"split/test.npy")
else:
    print("Doing new split...")
    test_id, train_id = MatTrain.train_test_indices(Nsamp, 0.2)
    np.save(cache_path+"split/train.npy", train_id)
    np.save(cache_path+"split/test.npy", test_id)
print("Done.")

print("Rescaling features...")
xscaler = MatTrain.UniformScaler()
xscaler.fit(cosmos[train_id])
trainx = xscaler.transform(cosmos[train_id])
testx = xscaler.transform(cosmos[test_id])
np.save(cache_path+"scalers/xscaler_min_diff",
        np.vstack([xscaler.min_val,xscaler.diff]))
print("Done.")


for component in args.to_train.split(" "):
    print("Loading {a} data...".format(a=component))
    P0_data = []
    P2_data = []
    data = [P0_data, P2_data]
    for i, l in enumerate(["0", "2"]):
        print(l)
        for file in sorted(os.listdir(inputY_path+component+l)):
            print(file)
            data[i].append(np.load(inputY_path+component+l+"/"+file))
    P0_data = np.vstack(data[0])
    P2_data = np.vstack(data[1])
    print("Done.")

    print("Flattening...")
    # Flatten all terms into one long vector.
    Ncomp = P0_data.shape[1]
    P0_data = P0_data[:,:,:39].reshape(Nsamp,Ncomp*39)
    P2_data = P2_data[:,:,:39].reshape(Nsamp,Ncomp*39)
    print("Done.")

    print("Removing zeros...")
    # Remove columns that are zero for all cosmos.
    P0_nonzero_cols = np.all(P0_data!=0.,axis=0)
    P2_nonzero_cols = np.all(P2_data!=0.,axis=0)
    P0_data = P0_data[:,P0_nonzero_cols]
    P2_data = P2_data[:,P2_nonzero_cols]
    np.save(cache_path+"scalers/{a}0/nonzero_cols".format(a=component),
            P0_nonzero_cols)
    np.save(cache_path+"scalers/{a}2/nonzero_cols".format(a=component),
            P2_nonzero_cols)
    print("Done.")


    print("Rescaling...")
    P0_scaler = MatTrain.UniformScaler()
    P2_scaler = MatTrain.UniformScaler()
    P0_scaler.fit(P0_data[train_id])
    P2_scaler.fit(P2_data[train_id])
    P0_trainy = P0_scaler.transform(P0_data[train_id])
    P2_trainy = P2_scaler.transform(P2_data[train_id])
    np.save(cache_path+"scalers/{a}0/yscaler_min_diff".format(a=component),
            np.vstack([P0_scaler.min_val,P0_scaler.diff]))
    np.save(cache_path+"scalers/{a}2/yscaler_min_diff".format(a=component),
            np.vstack([P2_scaler.min_val,P2_scaler.diff]))
    print("Done.")

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,
                                min_lr=0.0001, mode='min', cooldown=10)
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=20,
                            verbose=0, mode='min', baseline=None,
                            restore_best_weights=True)
    callbacks_list = [reduce_lr, early_stop]

    print("Training NNs...")
    P0_model = MatTrain.trainNN(trainx, P0_trainy, validation_data=None, nodes=np.array(arch_dict[component][0]),
                         learning_rate=0.013, batch_size=100, epochs=10000, callbacks=callbacks_list,
                         verbose=args.verbose)
    print("{a}{b} final loss: ".format(a=component, b=0), P0_model.evaluate(trainx, P0_trainy))

    P2_model = MatTrain.trainNN(trainx, P2_trainy, validation_data=None, nodes=np.array(arch_dict[component][1]),
                         learning_rate=0.013, batch_size=100, epochs=10000, callbacks=callbacks_list,
                         verbose=args.verbose)
    print("{a}{b} final loss: ".format(a=component, b=2), P2_model.evaluate(trainx, P2_trainy))

    P0_model.save(cache_path+"models/{a}0/member_0".format(a=component))
    P2_model.save(cache_path+"models/{a}2/member_0".format(a=component))
    print("Done.")

