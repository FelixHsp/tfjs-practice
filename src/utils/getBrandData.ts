export const getBrandData = async () => {
  const inputs = [];
  const labels = [];
  for (let i = 0; i < 30; i++) {
    ['android', 'apple', 'windows'].forEach(label => {
      const imageSrc = `../data/brand/train/${label}-${i}.jpg`;
      inputs.push(imageSrc);
      labels.push([
        label === 'android' ? 1 : 0,
        label === 'apple' ? 1 : 0,
        label === 'windows' ? 1 : 0,
      ]);
    });
  }
  
  return {
    inputs,
    labels,
  };
};