const trainGlobalModelBtn = document.getElementById('trainGlobalModelBtn');
const confirmTrainModal = document.getElementById('confirmTrainModal');
const trainingResultModal = document.getElementById('trainingResultModal');
const cancelTrainBtn = document.getElementById('cancelTrainBtn');
const confirmTrainBtn = document.getElementById('confirmTrainBtn');
const closeResultBtn = document.getElementById('closeResultBtn');

const showModal = (modal) => {
  modal.hidden = false;
};

const hideModal = (modal) => {
  modal.hidden = true;
};

const trainGlobalModel = () => {
  // Placeholder for API call.
  showModal(trainingResultModal);
};

trainGlobalModelBtn.addEventListener('click', () => {
  showModal(confirmTrainModal);
});

cancelTrainBtn.addEventListener('click', () => {
  hideModal(confirmTrainModal);
});

confirmTrainBtn.addEventListener('click', () => {
  hideModal(confirmTrainModal);
  trainGlobalModel();
});

closeResultBtn.addEventListener('click', () => {
  hideModal(trainingResultModal);
});
