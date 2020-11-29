import * as tf from '@tensorflow/tfjs-node';

const heights = [170, 180, 190];
const weights = [70, 80, 90];

const SGD = 0.1;
const BATCH_SIZE = 3;
const EPOCHS = 100;

const run = async () => {
  // 归一化: 压缩为0-1
  const inputs = tf.tensor(heights).sub(170).div(20);
  const labels = tf.tensor(weights).sub(70).div(20);

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

  await model.fit(inputs, labels, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS
  });

  const output: any = model.predict(tf.tensor([200]).sub(170).div(20));
  console.log(`predict: ${output.mul(20).add(70).dataSync()}`);
};

run();