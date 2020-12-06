export enum ELabelType {
  CHEW = 'CHEW',
  RUMINATION = 'RUMINATION'
}

export const LABEL_TYPE = {
  [ELabelType.CHEW]: {
    label: [1, 0],
    name: '进食咀嚼'
  },
  [ELabelType.RUMINATION]: {
    label: [0, 1],
    name: '反刍咀嚼'
  }
};