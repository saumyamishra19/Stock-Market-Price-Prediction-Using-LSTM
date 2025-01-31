from Core.data import Data
from Core.config import Column
from Core.model import model_train_seq_LSTM

data = Data()
data.read('data/BAJAJ-AUTO.csv')
data.check_null_values()
data.clean_dataset()
print(Column.OPEN.value)
data.print_head()
data.print_description()
data.normalize()
data.visualize_open(Column.OPEN.value)
data.visualize_close(Column.CLOSE.value)

trainer = model_train_seq_LSTM(data.dataframe, data.scaler)
trainer.build_train_lstm()
trainer.predict_plot()
trainer.evaluate_model()
trainer.save_model()
trainer.load_model()