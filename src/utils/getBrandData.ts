import { ELabelType, LABEL_TYPE } from '../constants';

const fs = require('fs');
const path = require('path');

export const getBrandData = async () => {
  const inputs = [];
  const labels = [];
  fs.readdirSync('../data/cow').forEach((file: string) => {
    if (/jpg/.test(file)) {
      const labelType = /rumination/.test(file) ? ELabelType.RUMINATION  : ELabelType.CHEW;
      const pathname = path.join('../data/cow', file);
      const label = LABEL_TYPE[labelType].label;
      console.log(`输入训练图片:${LABEL_TYPE[labelType].name}`);
      inputs.push(pathname);
      labels.push(label);
    }
  });

  return {
    inputs,
    labels,
  };
};