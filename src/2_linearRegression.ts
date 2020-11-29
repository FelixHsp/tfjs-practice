import * as tf from '@tensorflow/tfjs-node';

const xs: Array<number> = [1, 2, 3, 4];
const ys: Array<number> = [1, 3, 5, 7];

const SGD = 0.1;
const BATCH_SIZE = 1;
const EPOCHS = 100;

const run = async () => {
  // init model
  const model = tf.sequential();

  // add dense
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }));

  // 损失函数和优化器
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(SGD)
  });


  const inputs = tf.tensor(xs);
  const labels = tf.tensor(ys);
  
  await model.fit(inputs, labels, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS
  });

  const output = model.predict(tf.tensor([5]));
  console.log(`predict: ${(output as any).dataSync()}`);
};

run();