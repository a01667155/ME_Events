import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_video
from torchvision import transforms

class ResizeVideo:
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size)

    def __call__(self, video):
        C, T, H, W = video.shape
        resized_frames = [self.resize(video[:, t, :, :]) for t in range(T)]
        return torch.stack(resized_frames, dim=1)  # (C, T, H, W)

class CASME2_Events_Dataset(Dataset):
    def __init__(self):
        """
        Inicializa el dataset CASME II.
        :param dataframe: DataFrame de pandas con información de videos y etiquetas.
        :param video_folder: Ruta donde están almacenados los videos.
        :param transform: Transformaciones opcionales para los videos.
        """
        self.csv_path = 'D:\PythonCourse\ME_Recognition\Data\Labels_complete.csv'
        self.dataframe = pd.read_csv(self.csv_path)
        self.video_folder_path = r'D:\PythonCourse\ME_Recognition\Data\ME_Events'
        
        
        self.label_mapping = {label: idx for idx, label in enumerate(self.dataframe['Simple_label'].unique())}

    def __len__(self):
        """
        Devuelve el número total de muestras en el dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Obtiene una muestra (video y etiqueta) del dataset.
        :param index: Índice de la fila en el DataFrame.
        :return: Video procesado (como tensor) y su etiqueta.
        """
        video_path = self._get_video_path(index)

        print(f"Procesando video: {video_path}")

        # Cargar y procesar el video
        video = self._load_video(video_path)
        
        if self.transform:
            video = self.transform(video)
        
        # Rellenar el video para que coincida con la longitud máxima
        video = self._pad_video(video)

        # Convertir etiqueta a número
        label = self.label_mapping[self._get_label(index)]

        return video, torch.tensor(label, dtype=torch.long)

    def _get_video_path(self, index):
        """
        Obtiene la ruta completa de un video.
        """
        video_name = self.dataframe.iloc[index]['File_path']
        return f"{video_name}"

    def _get_label(self, index):
        """
        Obtiene la etiqueta de una muestra.
        """
        return self.dataframe.iloc[index]['Simple_label']

    def _load_video(self, video_path):
        """
        Método para cargar un video.
        Por ahora, lo dejaremos como un placeholder que puedes completar.
        """
        video, _, _ = read_video(video_path, pts_unit='sec')  # Tensor de forma (T, H, W, C)
        if video.shape[0] == 0:  # Si no tiene frames
            raise RuntimeError(f"El video en {video_path} no tiene frames válidos.")
        video = video.permute(3, 0, 1, 2)  # Convertir a (C, T, H, W)
        return video
    
    def _get_max_length(self):
        """
        Encuentra la longitud máxima (número de frames) en el dataset.
        """
        max_length = 0
        for idx in range(len(self.dataframe)):
            video_path = self._get_video_path(idx)
            video, _, _ = read_video(video_path, pts_unit='sec')
            max_length = max(max_length, video.shape[0])
        return max_length
    
    def _pad_video(self, video):
        """
        Rellena el video para que tenga la longitud máxima.
        """
        C, T, H, W = video.shape
        if T < self.max_length:
            pad_size = self.max_length - T
            padding = torch.zeros((C, pad_size, H, W))  # Frames vacíos
            video = torch.cat([video, padding], dim=1)
        return video

