import * as tf from '@tensorflow/tfjs-node';
import { getXorData } from './utils/getRandomData';

const data = getXorData(400);

const run = async () => {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'relu'
  }));
  
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));

  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1)
  });

  const inputs = tf.tensor(data.map(item => {
    return [item.x, item.y]
  }));
  const labels = tf.tensor(data.map(item => {
    return item.label
  }));

  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 100
  });

  const output: any = model.predict(tf.tensor([[-1, -1], [1, 1], [0, 0], [1, -1], [-1, 1]]));
  console.log(`predict: ${output.dataSync()}`);
};

run();