import * as tf from '@tensorflow/tfjs-node';

/*
const t0 = tf.tensor(1);
t0.print();
console.log(t0); // shape: []

const t1 = tf.tensor([1, 2]);
t1.print();
console.log(t1); // shape: [ 2 ]

const t2 = tf.tensor([[1, 2], [3, 4]]);
t2.print();
console.log(t2); // shape: [ 2, 2 ]

const t3 = tf.tensor([[[1]]]);
t3.print();
console.log(t3); // shape: [ 1, 1, 1 ]
*/


// for循环向量化
const input = [1, 2, 3, 4];
const weight = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];
const output = [0, 0, 0, 0];

for (let i = 0; i < weight.length; i++) {
  for (let j = 0; j < input.length; j++) {
    output[i] += input[j] * weight[i][j];
  }
}

console.log(output);

tf.tensor(weight).dot(tf.tensor(input)).print();