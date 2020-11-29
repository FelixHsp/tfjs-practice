import * as tf from '@tensorflow/tfjs-node';
import { getData } from './utils/getRandomData';

const data: Array<{ x: number, y: number, label: number }> = getData(400);

const run = async () => {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [2], // tf.tensor([x, y])
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
    epochs: 50
  });

  const output: any = model.predict(tf.tensor([[2, 1], [0, 0], [1, 2]]));
  console.log(`predict: ${output.dataSync()}`);
};

run();