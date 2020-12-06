import * as tf from '@tensorflow/tfjs-node';
import { getBrandData } from './utils/getBrandData';

const fs = require('fs');
const jpeg = require('jpeg-js');

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';
const NUM_CLASSES = 2;

const readImage = path => {
  const buf = fs.readFileSync(path);
  const pixels = jpeg.decode(buf, true);
  return pixels;
};

const imageByteArray = (image, numChannels) => {
  const pixels = image.data
  const numPixels = image.width * image.height;
  const values = new Int32Array(numPixels * numChannels);

  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel];
    }
  }

  return values;
};

const imageToInput = (image, numChannels) => {
  const values = imageByteArray(image, numChannels);
  const outShape: [number, number, number] = [image.height, image.width, numChannels];
  const input = tf.tensor3d(values, outShape, 'int32');

  return input;
};

const run = async () => {
  const { inputs, labels } = await getBrandData();

  const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  mobilenet.summary();
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  const truncatedMobilenet = tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output
  });

  const model = tf.sequential();
  model.add(tf.layers.flatten({
    // @ts-ignore
    inputShape: layer.outputShape.slice(1)
  }));
  model.add(tf.layers.dense({
    units: 10,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: NUM_CLASSES,
    activation: 'softmax'
  }));
  model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.adam() });

  const { xs, ys } = tf.tidy(() => {
    // @ts-ignore
    const xs = tf.concat(inputs.map(imgEl => {
      return truncatedMobilenet.predict(tf.tidy(() => {
        const image = readImage(imgEl)
        const input = imageToInput(image, 3);
        const imageResize = tf.image.resizeBilinear(
          input,
          [224, 224],
          false
        );
        const outputImage = imageResize
          .toFloat()
          .sub(255 / 2)
          .div(255 / 2)
          .reshape([1, 224, 224, 3])
        return outputImage;
      }));
    }));

    const ys = tf.tensor(labels);
    return { xs, ys };
  });

  await model.fit(xs, ys, {
    epochs: 500
  });

  const getPred: any = (file: string) => {
    return tf.tidy(() => {
      const input = truncatedMobilenet.predict(tf.tidy(() => {
        const image = readImage(file);
        const input = imageToInput(image, 3);
        const imageResize = tf.image.resizeBilinear(
          input,
          [224, 224],
          false
        );
        const outputImage = imageResize
          .toFloat()
          .sub(255 / 2)
          .div(255 / 2)
          .reshape([1, 224, 224, 3])
        return outputImage;
      }));
      return model.predict(input, {verbose: true});
    })
  };

  console.log(`预测结果：res1:${getPred('../data/1.jpg')}`);
  console.log(`预测结果：res2:${getPred('../data/2.jpg')}`);
  console.log(`预测结果：res3:${getPred('../data/3.jpg')}`);
  console.log(`预测结果：res4:${getPred('../data/4.jpg')}`);
};

run();