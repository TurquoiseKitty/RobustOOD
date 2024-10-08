
from .PTC import PTC_Dataset



def load_dataset(data_path, test_path, plot_path):
    """Loads the dataset."""

    dataset = PTC_Dataset(root=data_path,test=test_path,plot = plot_path)

        
    return dataset

if __name__=="main":
    dataset = load_dataset(data_path, test_path,plot_path)
