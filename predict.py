#!/usr/bin/env python3

import numpy as np
import pandas as pd
import onnxruntime as rt


def main():
    data = pd.read_csv("test.csv.gz", index_col="ID")
    x = np.asarray(data, dtype=np.float32)

    sess = rt.InferenceSession("particles.onnx")

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    print("Inputs: {}".format([x.name for x in sess.get_inputs()]))
    print("Outputs: {}".format([x.name for x in sess.get_outputs()]))

    pred = sess.run([label_name], {input_name: x})
    np.savetxt("predict.csv", pred[0], delimiter=",")

if __name__ == "__main__":
    main()

